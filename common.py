"""
common.py — código compartilhado por TODOS os experimentos (MLP e CNN).

Regras do grupo (README):
  - Todos usam o mesmo split.json (70/15/15, semente 42). Ninguém refaz a divisão.
  - Pesos de classe e aumento de dados só no treino.
  - A validação escolhe configurações; o teste só é olhado no final (03_resultados.py).
  - Ordem fixa das classes: glass, paper, cardboard, plastic, metal, trash (0 a 5).

Quem quiser mudar algo aqui avisa o grupo antes, porque afeta todos os modelos.
"""
import json
import os
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constantes do trabalho
# ---------------------------------------------------------------------------
SEED = 42
CLASSES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]
NUM_CLASSES = len(CLASSES)

# (altura, largura) — o TensorFlow espera altura × largura.
# As imagens originais têm 512 × 384 (largura × altura).
RESOLUCOES = {
    128: (96, 128),
    256: (192, 256),
    512: (384, 512),
}

# Os 6 experimentos do trabalho. Cada script de treino escolhe os seus.
EXPERIMENTOS = {
    "mlp_128_sem": {"modelo": "mlp", "res": 128, "pesos": False},
    "mlp_128_com": {"modelo": "mlp", "res": 128, "pesos": True},
    "cnn_128_sem": {"modelo": "cnn", "res": 128, "pesos": False},
    "cnn_128_com": {"modelo": "cnn", "res": 128, "pesos": True},
    "cnn_256_sem": {"modelo": "cnn", "res": 256, "pesos": False},
    "cnn_512_sem": {"modelo": "cnn", "res": 512, "pesos": False},
}

# Hiperparâmetros de treino comuns (iguais para todos, para a comparação ser justa)
TREINO_PADRAO = {
    "batch_size": 32,
    "epocas_max": 100,
    "paciencia_early_stopping": 10,   # em val_loss
    "paciencia_reduce_lr": 4,
    "fator_reduce_lr": 0.5,
    "lr_minimo": 1e-6,
}

# Pastas de saída
DIR_MODELOS = Path("modelos")
DIR_HISTORICOS = Path("historicos")
DIR_CURVAS = Path("figuras/curvas")
DIR_CHECKPOINTS = Path("checkpoints")


# ---------------------------------------------------------------------------
# Semente
# ---------------------------------------------------------------------------
def fixar_semente(seed: int = SEED, deterministico: bool = False):
    """Fixa as sementes do Python, NumPy e TensorFlow.

    deterministico=True deixa as operações da GPU determinísticas (mais lento e
    algumas operações podem dar erro). Por padrão fica desligado.
    """
    import random
    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    if deterministico:
        tf.config.experimental.enable_op_determinism()


# ---------------------------------------------------------------------------
# split.json
# ---------------------------------------------------------------------------
def carregar_split(caminho: str = "split.json") -> dict:
    split = json.loads(Path(caminho).read_text(encoding="utf-8"))
    if split["classes"] != CLASSES:
        raise ValueError(f"Ordem de classes do split.json difere da combinada: {split['classes']}")
    if split["seed"] != SEED:
        raise ValueError(f"split.json foi gerado com semente {split['seed']}, esperado {SEED}")
    return split


def caminhos_e_rotulos(split: dict, conjunto: str, data_dir: str = "dataset-resized"):
    """Devolve (lista de caminhos completos, array de rótulos 0..5) de 'train', 'val' ou 'test'."""
    itens = split[conjunto]
    caminhos = [str(Path(data_dir) / it["path"]) for it in itens]
    rotulos = np.array([CLASSES.index(it["label"]) for it in itens], dtype=np.int32)
    faltando = [c for c in caminhos if not Path(c).exists()]
    if faltando:
        raise FileNotFoundError(
            f"{len(faltando)} imagens de '{conjunto}' não encontradas (ex.: {faltando[0]}). "
            f"Confira se a pasta '{data_dir}/' está na raiz do projeto.")
    return caminhos, rotulos


def pesos_de_classe(split: dict) -> dict:
    """Pesos 'balanced' calculados só no treino (já estão no split.json), no formato do Keras: {0: w, 1: w, ...}."""
    return {i: float(split["class_weights"][c]) for i, c in enumerate(CLASSES)}


# ---------------------------------------------------------------------------
# Pipeline de dados (tf.data)
# ---------------------------------------------------------------------------
def _carregar_imagem(caminho, rotulo, altura, largura):
    import tensorflow as tf

    img = tf.io.read_file(caminho)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (altura, largura), antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)  # cache em uint8 gasta 4× menos RAM
    return img, rotulo


def _aumentar(img, rotulo):
    """Aumento de dados leve — SÓ no treino. Igual para MLP e CNN."""
    import tensorflow as tf

    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    return tf.clip_by_value(img, 0.0, 1.0), rotulo


def _normalizar(img, rotulo):
    import tensorflow as tf

    return tf.cast(img, tf.float32) / 255.0, rotulo


def criar_dataset(split: dict, conjunto: str, res: int, batch_size: int = 32,
                  data_dir: str = "dataset-resized", aumento: bool = None, cache: bool = True):
    """Cria o tf.data.Dataset de 'train', 'val' ou 'test' na resolução pedida (128, 256 ou 512).

    - Pixels normalizados para [0, 1].
    - Treino: embaralhado (semente 42) e com aumento de dados.
    - Val/teste: ordem fixa, sem aumento.
    """
    import tensorflow as tf

    altura, largura = RESOLUCOES[res]
    treino = conjunto == "train"
    if aumento is None:
        aumento = treino
    if aumento and not treino:
        raise ValueError("Aumento de dados só pode ser usado no treino.")

    caminhos, rotulos = caminhos_e_rotulos(split, conjunto, data_dir)
    ds = tf.data.Dataset.from_tensor_slices((caminhos, rotulos))
    ds = ds.map(lambda c, r: _carregar_imagem(c, r, altura, largura),
                num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    if treino:
        ds = ds.shuffle(len(caminhos), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_normalizar, num_parallel_calls=tf.data.AUTOTUNE)
    if aumento:
        ds = ds.map(_aumentar, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Treino padronizado (com retomada se o Colab cair)
# ---------------------------------------------------------------------------
def ja_treinado(nome: str) -> bool:
    return (DIR_MODELOS / f"{nome}.keras").exists() and (DIR_HISTORICOS / f"{nome}.json").exists()


def _ler_historico_csv(caminho) -> dict:
    """Lê o CSV de épocas; se uma época aparece duas vezes (retomada), fica a última."""
    import csv

    linhas = {}
    with open(caminho, newline="", encoding="utf-8") as f:
        for linha in csv.DictReader(f):
            linhas[int(linha["epoch"])] = linha
    historico = {}
    for ep in sorted(linhas):
        for k, v in linhas[ep].items():
            if k == "epoch" or v in (None, ""):
                continue
            historico.setdefault(k, []).append(float(v))
    return historico


def _TempoPorEpoca():
    """Callback que mede o tempo de cada época e grava no log (vai para o CSV)."""
    import tensorflow as tf

    class TempoPorEpoca(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self._t0 = time.time()

        def on_epoch_end(self, epoch, logs=None):
            if logs is not None:
                logs["tempo_epoca_s"] = time.time() - self._t0

    return TempoPorEpoca()


def _memoria_gpu_pico_mb():
    import tensorflow as tf

    try:
        if tf.config.list_physical_devices("GPU"):
            return round(tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2**20, 1)
    except Exception:
        pass
    return None


def _memoria_ram_pico_mb():
    """Pico de RAM do processo (Linux/Colab: ru_maxrss vem em KB)."""
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    except Exception:
        return None


def _hardware():
    import platform
    import tensorflow as tf

    gpus = []
    for g in tf.config.list_physical_devices("GPU"):
        try:
            gpus.append(tf.config.experimental.get_device_details(g).get("device_name", g.name))
        except Exception:
            gpus.append(g.name)
    return {"gpu": gpus or None, "cpu": platform.processor() or platform.machine(),
            "tensorflow": tf.__version__, "python": platform.python_version()}


def f1_macro_validacao(modelo, split: dict, res: int, batch_size: int = 32,
                       data_dir: str = "dataset-resized"):
    """F1 macro na VALIDAÇÃO — métrica principal de seleção pedida no enunciado."""
    from sklearn.metrics import accuracy_score, f1_score

    ds_val = criar_dataset(split, "val", res, batch_size, data_dir)
    _, y_true = caminhos_e_rotulos(split, "val", data_dir)
    y_pred = np.argmax(modelo.predict(ds_val, verbose=0), axis=1)
    return (float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            float(accuracy_score(y_true, y_pred)))


def treinar_experimento(nome: str, construir_modelo, lr: float, split_path: str = "split.json",
                        data_dir: str = "dataset-resized", batch_size: int = None,
                        epocas_max: int = None, forcar: bool = False, info_extra: dict = None):
    """Treina um dos 6 experimentos de forma padronizada.

    construir_modelo(input_shape, num_classes) -> tf.keras.Model (sem compilar).

    Salva:
      modelos/<nome>.keras          melhor modelo (menor val_loss), SEM o estado do otimizador
      historicos/<nome>.json        histórico por época + metadados (custo, F1 macro de validação...)
      historicos/<nome>_epocas.csv  log bruto por época
      figuras/curvas/<nome>.png     curvas de perda e acurácia (treino × validação)

    Se o experimento já terminou, pula (a não ser que forcar=True).
    Se o Colab cair no meio, roda de novo que ele continua da última época salva.

    Medição de tempo: relógio de parede de cada época de model.fit (treino + validação),
    EXCLUINDO download do dataset e a leitura/redimensionamento inicial das imagens
    (feita antes, no aquecimento do cache, e registrada à parte).
    """
    import shutil
    import tensorflow as tf

    if nome not in EXPERIMENTOS:
        raise ValueError(f"Experimento desconhecido: {nome}")
    if ja_treinado(nome) and not forcar:
        print(f"[{nome}] já treinado — pulando. (use --forcar para treinar de novo)")
        return None

    cfg = EXPERIMENTOS[nome]
    batch_size = batch_size or TREINO_PADRAO["batch_size"]
    epocas_max = epocas_max or TREINO_PADRAO["epocas_max"]
    for d in (DIR_MODELOS, DIR_HISTORICOS, DIR_CURVAS, DIR_CHECKPOINTS):
        d.mkdir(parents=True, exist_ok=True)

    caminho_modelo = DIR_MODELOS / f"{nome}.keras"
    caminho_melhor = DIR_CHECKPOINTS / f"{nome}_melhor.keras"   # com otimizador, só para retomar
    caminho_csv = DIR_HISTORICOS / f"{nome}_epocas.csv"
    if forcar:  # recomeça do zero, com pesos novos
        shutil.rmtree(DIR_CHECKPOINTS / nome, ignore_errors=True)
        for p in (caminho_csv, caminho_melhor, caminho_modelo):
            p.unlink(missing_ok=True)

    fixar_semente()
    split = carregar_split(split_path)
    ds_treino = criar_dataset(split, "train", cfg["res"], batch_size, data_dir)
    ds_val = criar_dataset(split, "val", cfg["res"], batch_size, data_dir)

    # Aquecimento: lê e redimensiona todas as imagens uma vez (preenche o cache),
    # para que esse preparo não entre no tempo de treino.
    t0 = time.time()
    for _ in ds_treino:
        pass
    for _ in ds_val:
        pass
    tempo_preparo = time.time() - t0
    fixar_semente()  # o aquecimento consumiu números aleatórios; volta ao estado inicial

    altura, largura = RESOLUCOES[cfg["res"]]
    modelo = construir_modelo((altura, largura, 3), NUM_CLASSES)
    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
    modelo.summary()

    pesos = pesos_de_classe(split) if cfg["pesos"] else None
    print(f"[{nome}] resolução {largura}×{altura} | lote {batch_size} | pesos de classe: {pesos}")

    # Se estiver retomando, o checkpoint só deve sobrescrever o melhor modelo se melhorar
    # em relação às épocas já feitas antes da queda.
    limiar = None
    if caminho_csv.exists():
        try:
            limiar = min(_ler_historico_csv(caminho_csv)["val_loss"])
            print(f"[{nome}] retomando; melhor val_loss até agora: {limiar:.4f}")
        except Exception:
            pass

    callbacks = [
        tf.keras.callbacks.BackupAndRestore(str(DIR_CHECKPOINTS / nome)),
        _TempoPorEpoca(),
        tf.keras.callbacks.ModelCheckpoint(str(caminho_melhor), monitor="val_loss",
                                           save_best_only=True, initial_value_threshold=limiar),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                         patience=TREINO_PADRAO["paciencia_early_stopping"],
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                             factor=TREINO_PADRAO["fator_reduce_lr"],
                                             patience=TREINO_PADRAO["paciencia_reduce_lr"],
                                             min_lr=TREINO_PADRAO["lr_minimo"]),
        tf.keras.callbacks.CSVLogger(str(caminho_csv), append=True),
    ]

    # Caso raro: o fit já terminou mas o Colab caiu antes de salvar os resultados
    fit_concluido = (caminho_melhor.exists() and caminho_csv.exists()
                     and not (DIR_CHECKPOINTS / nome).exists())
    inicio = time.time()
    if fit_concluido:
        print(f"[{nome}] treino já concluído antes; só gerando os resultados.")
    else:
        modelo.fit(ds_treino, validation_data=ds_val, epochs=epocas_max,
                   class_weight=pesos, callbacks=callbacks, verbose=2)
    tempo_sessao = time.time() - inicio

    # O histórico vem do CSV, que guarda TODAS as épocas mesmo se o Colab caiu no meio
    historico = _ler_historico_csv(caminho_csv)
    tempos = historico.get("tempo_epoca_s")
    tempo_total = sum(tempos) if tempos else tempo_sessao
    epocas_rodadas = len(historico["val_loss"])
    melhor_epoca = int(np.argmin(historico["val_loss"])) + 1

    # Modelo final = melhor época em val_loss, salvo sem otimizador (arquivo menor)
    melhor = tf.keras.models.load_model(caminho_melhor, compile=False)
    melhor.save(caminho_modelo)
    tamanho_mb = round(caminho_modelo.stat().st_size / 2**20, 2)

    f1_val, acc_val = f1_macro_validacao(melhor, split, cfg["res"], batch_size, data_dir)

    meta = {
        "experimento": nome,
        **cfg,
        "entrada_altura_largura_canais": [altura, largura, 3],
        "normalizacao": "pixels / 255 (RGB)",
        "aumento_de_dados_treino": "flip horizontal e vertical, brilho ±0,1, contraste 0,9–1,1",
        "funcao_de_perda": "sparse_categorical_crossentropy" + (" ponderada por classe" if pesos else ""),
        "pesos_de_classe": pesos,
        "otimizador": "Adam",
        "lr_inicial": lr,
        "batch_size": batch_size,
        "criterio_parada": (f"EarlyStopping em val_loss, paciência {TREINO_PADRAO['paciencia_early_stopping']}, "
                            f"restaurando os melhores pesos; ReduceLROnPlateau fator "
                            f"{TREINO_PADRAO['fator_reduce_lr']}, paciência {TREINO_PADRAO['paciencia_reduce_lr']}"),
        "epocas_max": epocas_max,
        "epocas_rodadas": epocas_rodadas,
        "epoca_do_modelo_salvo": melhor_epoca,
        "semente": SEED,
        "melhor_val_loss": min(historico["val_loss"]),
        "val_accuracy": acc_val,
        "val_f1_macro": f1_val,
        "parametros_treinaveis": int(sum(np.prod(w.shape) for w in melhor.trainable_weights)),
        "parametros_totais": int(melhor.count_params()),
        "tamanho_modelo_salvo_mb": tamanho_mb,
        "tempo_total_treino_s": round(tempo_total, 1),
        "tempo_medio_por_epoca_s": round(tempo_total / epocas_rodadas, 2),
        "tempo_preparo_cache_s_excluido": round(tempo_preparo, 1),
        "metodo_medicao_tempo": "time.time() no início e fim de cada época do model.fit (treino + validação); "
                                "exclui download e leitura/redimensionamento inicial das imagens",
        "memoria_gpu_pico_mb": _memoria_gpu_pico_mb(),
        "memoria_ram_pico_mb": _memoria_ram_pico_mb(),
        "hardware": _hardware(),
    }
    if info_extra:
        meta.update(info_extra)

    (DIR_HISTORICOS / f"{nome}.json").write_text(
        json.dumps({"meta": meta, "historico": historico}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    plotar_curvas(historico, nome, DIR_CURVAS / f"{nome}.png", melhor_epoca)

    print(f"\n[{nome}] fim: {epocas_rodadas} épocas, modelo salvo da época {melhor_epoca}, "
          f"val_F1_macro {f1_val:.3f}, val_acc {acc_val:.3f}, "
          f"{tempo_total / 60:.1f} min ({meta['tempo_medio_por_epoca_s']:.1f} s/época), {tamanho_mb} MB")
    return meta


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def plotar_curvas(historico: dict, titulo: str, saida, melhor_epoca: int = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epocas = np.arange(1, len(historico["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, chave, rotulo in [(axes[0], "loss", "Perda"), (axes[1], "accuracy", "Acurácia")]:
        ax.plot(epocas, historico[chave], label="treino")
        ax.plot(epocas, historico[f"val_{chave}"], label="validação")
        if melhor_epoca:
            ax.axvline(melhor_epoca, color="gray", ls="--", lw=1, label=f"melhor época ({melhor_epoca})")
        ax.set_xlabel("Época")
        ax.set_ylabel(rotulo)
        ax.legend()
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(saida, dpi=150)
    plt.close(fig)


def carregar_historico(nome: str) -> dict:
    return json.loads((DIR_HISTORICOS / f"{nome}.json").read_text(encoding="utf-8"))
