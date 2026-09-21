"""finetune_tr.py — Auto-AVSR'ı Türkçe veriyle fine-tune eden hafif sürücü.

auto_avsr_train/train.py'nin YERİNE kullanılır çünkü o script SLURM cluster'ı
için yazılmış (SLURM_JOB_ID env var zorunlu, accelerator="gpu" sabit, wandb
hesabı gerektiren WandbLogger, DDPStrategy). Burada aynı ModelModule ve
DataModule (lightning.py, datamodule/) DEĞİŞTİRİLMEDEN kullanılıyor — sadece
etrafındaki sürücü basitleştirildi: CPU/MPS/GPU'da tek cihazla çalışır, wandb
veya SLURM gerektirmez.

Kullanım (kuru deneme, ücretsiz):
    .venv/bin/python finetune_tr.py --root-dir data --train-file trdry_train_transcript_lengths_seg6s.csv \
        --val-file trdry_val_transcript_lengths_seg6s.csv --max-epochs 1 --accelerator cpu

Kullanım (gerçek sıhhat testi, Modal GPU'da):
    python finetune_tr.py --root-dir /data --train-file tr_train_transcript_lengths_seg6s.csv \
        --val-file tr_val_transcript_lengths_seg6s.csv --max-epochs 5 --accelerator gpu \
        --pretrained-model-path vsr_trlrs3_base.pth --transfer-encoder
"""
import argparse
import sys

import torch
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENDOR_DIR = ROOT / "vendor" / "auto_avsr_train"
sys.path.insert(0, str(VENDOR_DIR))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-dir", required=True, help="labels/ ve master/ klasörlerinin bulunduğu kök dizin")
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--val-file", required=True)
    ap.add_argument("--test-file", default=None)
    ap.add_argument("--exp-dir", default=str(ROOT / "exp"))
    ap.add_argument("--exp-name", default="tr_finetune")
    ap.add_argument("--modality", default="video", choices=["video", "audio"])
    ap.add_argument("--max-epochs", type=int, default=1)
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.03)
    ap.add_argument("--ctc-weight", type=float, default=0.1)
    ap.add_argument("--max-frames", type=int, default=1600)
    ap.add_argument("--train-num-buckets", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=0, help="DataLoader worker sayısı (macOS'ta çoklu worker pickling sorunu çıkarabiliyor, kuru denemede 0 tut)")
    ap.add_argument("--pretrained-model-path", default=None)
    ap.add_argument("--transfer-frontend", action="store_true")
    ap.add_argument("--transfer-encoder", action="store_true")
    ap.add_argument("--accelerator", default="cpu", choices=["cpu", "mps", "gpu", "auto"])
    ap.add_argument("--devices", default=1)
    ap.add_argument("--checkpoint", action="store_true", help="epoch checkpoint'lerini kaydet (kuru denemede kapalı)")
    ap.add_argument("--ckpt-every-n-steps", type=int, default=0,
                    help="epoch BEKLEMEDEN her N adımda checkpoint yaz. Spot/preemptible "
                         "GPU'da şart: epoch 3-6 saat sürüyor, 0 ise preemption o saatleri çöpe atar.")
    ap.add_argument("--decode-ctc-weight", type=float, default=0.5,
                    help="COZUMLEME sirasindaki CTC agirligi (egitimdeki "
                         "--ctc-weight'ten AYRI). Auto-AVSR varsayilani 0.1 ve "
                         "tekrar dongulerine yol aciyor; olculdu 2026-09-04: "
                         "0.1 -> WER %%92, dongu 17/150 · 0.5 -> WER %%86, dongu 6/150. "
                         "0.7+ dongu azaltiyor ama attention'in dil modeli "
                         "katkisi kayboldugu icin WER geri kotulesiyor.")
    ap.add_argument("--test", action="store_true",
                    help="egitim bitince test split'inde WER olc. --test-file "
                         "verilmezse val kullanilir. LIGHTNING'de test_step VAR "
                         "ama finetune_tr.py onu hic cagirmiyordu -> 36 saat "
                         "kosup sayisiz kalinabilirdi.")
    ap.add_argument("--kac-eniyi", type=int, default=2,
                    help="loss_val'a gore kac checkpoint saklansin. >2 vermek "
                         "AGIRLIK ORTALAMASI icin gerekli (vendor/average_checkpoints.py); "
                         "ESPnet'te standart ve genelde ~1 puan bedava. Her biri 2,9 GB.")
    ap.add_argument("--flip", action="store_true",
                    help="egitimde yatay cevirme artirmasi (p=0.5). Dudak okumada "
                         "standart ama bu Auto-AVSR surumunde YOK. Bayrak olarak "
                         "eklendi ki etkisi OLCULEBILSIN, sessizce degismesin.")
    ap.add_argument("--probe", type=int, default=0,
                    help="N adim kosup tepe VRAM + adim suresini bildir, sonra dur. "
                         "GPU secmeden once --max-frames'in tavanini bulmak icin.")
    ap.add_argument("--resume", default=None,
                    help="checkpoint'ten devam et. Yol ver, ya da 'auto' de -> exp_dir/exp_name/last.ckpt")
    return ap.parse_args()


def main():
    cli = parse_args()

    # ModelModule / DataModule bu isimlerle args bekliyor (lightning.py, datamodule/data_module.py)
    # NOT: pytorch_lightning'in save_hyperparameters'ı yalnız argparse.Namespace
    # (ya da dict/dataclass) kabul ediyor — SimpleNamespace'i reddediyor.
    args = Namespace(
        root_dir=cli.root_dir,
        train_file=cli.train_file,
        val_file=cli.val_file,
        test_file=cli.test_file or cli.val_file,
        modality=cli.modality,
        max_epochs=cli.max_epochs,
        # ponytail: --probe max_epochs'i 1'e cekiyor; warmup varsayilani da 1
        # olunca cosine.py'de decay_steps = toplam - warmup = 0 -> ZeroDivisionError
        # (olculdu 2026-09-03, max_frames=1800). Olcumde LR programi zaten
        # anlamsiz, warmup'i kapat.
        warmup_epochs=0 if cli.probe else cli.warmup_epochs,
        lr=cli.lr,
        weight_decay=cli.weight_decay,
        ctc_weight=cli.ctc_weight,
        max_frames=cli.max_frames,
        pretrained_model_path=cli.pretrained_model_path,
        transfer_frontend=cli.transfer_frontend,
        transfer_encoder=cli.transfer_encoder,
        decode_snr_target=None,
    )

    from lightning import ModelModule
    from datamodule.data_module import DataModule

    if cli.flip:
        # ponytail: vendor/ IC ICE GIT DEPOSU -- oraya yazilan degisiklik
        # commit EDILEMEZ ve tek laptopta kalir (bkz. Nerede-Ne-Var.md).
        # O yuzden dosyayi duzenlemek yerine calisma aninda sarmaliyoruz,
        # beam search'te yaptigimizin aynisi.
        #
        # Yatay cevirme dudak okumada standart artirma ve bu surumde YOK.
        # RandomHorizontalFlip 4B tensore (T,C,H,W) tek zar atar, yani tum
        # klip AYNI yone cevrilir -- kare kare cevrilseydi hareket bozulurdu.
        # torch'u BURADA yerel ada bagla: main() icinde asagida bir
        # "import torch" var ve o, kapanistaki `torch`u serbest degiskene
        # cevirip NameError attiriyordu (olculdu: egitim ilk adimda dustu).
        import torch as _torch
        import torchvision
        from datamodule import transforms as _tf
        _eski_init = _tf.VideoTransform.__init__

        def _yeni_init(self, subset):
            _eski_init(self, subset)
            if subset == "train":
                kat = list(self.video_pipeline)
                kat.insert(2, torchvision.transforms.RandomHorizontalFlip(0.5))
                self.video_pipeline = _torch.nn.Sequential(*kat)

        _tf.VideoTransform.__init__ = _yeni_init
        print("artirma: yatay cevirme AÇIK (p=0.5)")
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.loggers import CSVLogger

    seed_everything(42, workers=True)

    print(f"Türkçe vocab boyutu (token_list): kontrol ediliyor...")
    modelmodule = ModelModule(args)
    print(f"vocab boyutu: {len(modelmodule.token_list)}")

    datamodule = DataModule(args, train_num_buckets=cli.train_num_buckets, num_workers=cli.num_workers)

    callbacks = []
    if cli.checkpoint:
        from pytorch_lightning.callbacks import ModelCheckpoint
        # IKI AYRI checkpoint, iki ayri is:
        #
        # 1) DEVAM ETME. every_n_train_steps olmadan checkpoint yalniz epoch
        #    sonunda yaziliyor; 131 saatlik veride epoch ~1 saat, spot
        #    preemption o saati sifirlar. last.ckpt resume hedefi.
        callbacks.append(ModelCheckpoint(
            dirpath=str(Path(cli.exp_dir) / cli.exp_name),
            monitor="monitoring_step", mode="max", save_last=True,
            filename="devam-{epoch}-{step}", save_top_k=2,
            every_n_train_steps=cli.ckpt_every_n_steps or None,
        ))
        # 2) MODEL SECIMI. ponytail: yukaridaki "monitoring_step" adim
        #    numarasi -- mode="max" ile "en iyi" = "en son" demek. Yani
        #    save_top_k en iyi modelleri DEGIL en yeni checkpointleri tutuyor
        #    (Auto-AVSR varsayilani; amaci model secimi degil kosuyu surdurmek).
        #    Olculdu 2026-09-04: val_loss epoch 8'de dibe vurdu, egitim 19'a
        #    kadar surdu, epoch 8 ve 12'nin agirliklari SILINDI -- geri donusu
        #    yok. Overfitting'in en iyi anini kaybetmemek icin ayri, val_loss
        #    izleyen bir checkpoint gerekiyor.
        callbacks.append(ModelCheckpoint(
            dirpath=str(Path(cli.exp_dir) / cli.exp_name),
            monitor="loss_val", mode="min", save_top_k=cli.kac_eniyi,
            filename="eniyi-{epoch}-{loss_val:.2f}",
        ))

    trainer = Trainer(
        default_root_dir=cli.exp_dir,
        max_epochs=1 if cli.probe else cli.max_epochs,
        limit_train_batches=cli.probe or 1.0,
        limit_val_batches=0 if cli.probe else 1.0,
        accelerator=cli.accelerator,
        devices=cli.devices,
        callbacks=callbacks,
        logger=CSVLogger(save_dir=cli.exp_dir, name=cli.exp_name),
        gradient_clip_val=10.0,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
    )
    # ponytail: ckpt_path olmadan resume diye bir sey yok -- dusen spot instance
    # sifirdan basliyordu. 'auto' en son last.ckpt'yi bulur.
    devam = cli.resume
    if devam == "auto":
        aday = Path(cli.exp_dir) / cli.exp_name / "last.ckpt"
        devam = str(aday) if aday.exists() else None
        print(f"resume=auto -> {devam or 'checkpoint yok, sifirdan basliyor'}")
    if devam and not Path(devam).exists():
        raise SystemExit(f"checkpoint bulunamadi: {devam}")

    import time
    t0 = time.time()
    trainer.fit(model=modelmodule, datamodule=datamodule, ckpt_path=devam)
    gecen = time.time() - t0
    print(f"Egitim tamamlandi. son global_step={trainer.global_step}")

    if cli.test and not cli.probe:
        print("\n=== TEST (WER) ===")
        # ponytail: ckpt_path="best" ILK ModelCheckpoint'i baz aliyor, o da
        # adim numarasina bakiyor -> "en son". val_loss'a gore en iyiyi acikca
        # sec; yoksa overfit olmus son modeli olcmus oluruz.
        # ponytail: METINSEL siralama yanlis checkpoint seciyordu -- dosya adi
        # "eniyi-epoch=9-loss_val=88.89" ve alfabetik sirada "epoch=10" once
        # geliyor. Olculdu 2026-09-06: en iyi (88,89) yerine 89,03 test edildi.
        # loss_val'i ADINDAN SAYI olarak oku ve en kucugu sec.
        import re as _re
        adaylar = list(Path(cli.exp_dir).joinpath(cli.exp_name).glob("eniyi-*.ckpt"))

        def _loss(yol_):
            m = _re.search(r"loss_val=(\d+\.\d+)", yol_.stem)
            return float(m.group(1)) if m else float("inf")

        eniyi = sorted(adaylar, key=_loss)
        yol = str(eniyi[0]) if eniyi else "best"
        print(f"test edilen checkpoint: {yol}")
        # ponytail: get_beam_search_decoder lightning.py'de ctc_weight=0.1 ile
        # cagriliyor ve vendor/ ic ice git deposu -- oraya dokunmadan, decoder'i
        # test basladiktan SONRA degistiriyoruz.
        from lightning import get_beam_search_decoder
        _orij = modelmodule.on_test_epoch_start
        def _yeni():
            _orij()
            modelmodule.beam_search = get_beam_search_decoder(
                modelmodule.model, modelmodule.token_list,
                ctc_weight=cli.decode_ctc_weight)
            print(f"cozumleme ctc_weight={cli.decode_ctc_weight}")
        modelmodule.on_test_epoch_start = _yeni
        sonuc = trainer.test(model=modelmodule, datamodule=datamodule, ckpt_path=yol)
        for d in sonuc:
            for k, v in d.items():
                print(f"TEST {k}={v:.4f}")

    if cli.probe:
        import torch
        adim = trainer.global_step or 1
        print(f"\nPROBE max_frames={cli.max_frames} adim={adim} "
              f"sn/adim={gecen/adim:.2f}")
        if torch.cuda.is_available():
            tepe = torch.cuda.max_memory_allocated() / 2**30
            toplam = torch.cuda.get_device_properties(0).total_memory / 2**30
            print(f"PROBE tepe_VRAM={tepe:.1f} GiB / {toplam:.1f} GiB "
                  f"(%{100*tepe/toplam:.0f})")
            print(f"PROBE GPU={torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
