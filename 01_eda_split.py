"""
01_eda_split.py — Dados e EDA (TrashNet)

Faz:
  1. Confere integridade (abre cada imagem, checa tamanho 512x384 e canais de cor,
     detecta duplicatas por hash)
  2. Contagem por classe
  3. Divisão estratificada 70/15/15 com semente 42 -> split.json
     As duplicatas exatas ficam TODAS no treino, para que uma mesma imagem nunca
     apareça em dois conjuntos (evita vazamento para validação/teste).
  4. Pesos de classe (calculados SÓ no treino) -> dentro do split.json
  5. Figuras da EDA -> figuras/eda/

Uso:
  python 01_eda_split.py --data_dir dataset-resized
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

CLASSES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]  # ordem fixa do trabalho
SEED = 42
EXPECTED_SIZE = (512, 384)  # (largura, altura)
RESOLUCOES = [(128, 96), (256, 192), (512, 384)]  # (largura, altura)
EXTS = {".jpg", ".jpeg", ".png"}


def listar_imagens(data_dir: Path):
    paths, labels = [], []
    for c in CLASSES:
        pasta = data_dir / c
        if not pasta.is_dir():
            raise FileNotFoundError(f"Pasta da classe não encontrada: {pasta}")
        for p in sorted(pasta.iterdir()):
            if p.suffix.lower() in EXTS:
                paths.append(p)
                labels.append(c)
    return paths, labels


def checar_integridade(paths):
    corrompidas, tamanhos, modos, hashes = [], Counter(), Counter(), defaultdict(list)
    for p in paths:
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                tamanhos[im.size] += 1
                modos[im.mode] += 1
        except Exception as e:
            corrompidas.append((str(p), str(e)))
            continue
        hashes[hashlib.md5(p.read_bytes()).hexdigest()].append(str(p))
    duplicatas = [v for v in hashes.values() if len(v) > 1]
    return corrompidas, tamanhos, modos, duplicatas


def fig_distribuicao(contagem, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = [contagem[c] for c in CLASSES]
    bars = ax.bar(CLASSES, vals, color="#4C72B0")
    ax.bar_label(bars)
    ax.set_ylabel("Nº de imagens")
    ax.set_title("Distribuição de classes — TrashNet")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_distribuicao_split(split_counts, out):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(CLASSES))
    w = 0.27
    for i, nome in enumerate(["train", "val", "test"]):
        vals = [split_counts[nome][c] for c in CLASSES]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=nome)
        ax.bar_label(bars, fontsize=7)
    ax.set_xticks(x, CLASSES)
    ax.set_ylabel("Nº de imagens")
    ax.set_title("Distribuição por conjunto (70/15/15 estratificado)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_amostras(paths, labels, out, n=5):
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(len(CLASSES), n, figsize=(n * 2.2, len(CLASSES) * 1.8))
    for i, c in enumerate(CLASSES):
        idx = [k for k, l in enumerate(labels) if l == c]
        escolhidos = rng.choice(idx, size=min(n, len(idx)), replace=False)
        for j in range(n):
            ax = axes[i, j]
            ax.axis("off")
            if j < len(escolhidos):
                ax.imshow(Image.open(paths[escolhidos[j]]).convert("RGB"))
            if j == 0:
                ax.set_title(c, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def fig_resolucoes(caminho, out):
    """Uma mesma imagem nas três resoluções dos experimentos (exigido no enunciado)."""
    original = Image.open(caminho).convert("RGB")
    fig, axes = plt.subplots(1, len(RESOLUCOES), figsize=(12, 3.4))
    for ax, (w, h) in zip(axes, RESOLUCOES):
        ax.imshow(original.resize((w, h), Image.BILINEAR))
        ax.set_title(f"{w} × {h}")
        ax.axis("off")
    fig.suptitle(f"Mesma imagem nas três resoluções ({Path(caminho).name})")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_duplicatas(duplicatas, data_dir, out):
    """Mostra os pares de imagens idênticas com rótulos diferentes."""
    if not duplicatas:
        return
    n = max(len(g) for g in duplicatas)
    fig, axes = plt.subplots(len(duplicatas), n, figsize=(n * 2.6, len(duplicatas) * 2.2),
                             squeeze=False)
    for i, grupo in enumerate(duplicatas):
        for j in range(n):
            ax = axes[i, j]
            ax.axis("off")
            if j < len(grupo):
                ax.imshow(Image.open(grupo[j]).convert("RGB"))
                ax.set_title(str(Path(grupo[j]).relative_to(data_dir)), fontsize=8)
    fig.suptitle("Duplicatas exatas (mantidas juntas no treino)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def fig_intensidade(paths, labels, out, amostra_por_classe=80):
    """Média de R, G, B por classe (amostra) — ajuda a discutir se a cor separa as classes."""
    rng = np.random.default_rng(SEED)
    medias = {}
    for c in CLASSES:
        idx = [k for k, l in enumerate(labels) if l == c]
        idx = rng.choice(idx, size=min(amostra_por_classe, len(idx)), replace=False)
        rgb = [np.asarray(Image.open(paths[k]).convert("RGB").resize((128, 96))).reshape(-1, 3).mean(0)
               for k in idx]
        medias[c] = np.mean(rgb, axis=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(CLASSES))
    for i, (canal, cor) in enumerate(zip("RGB", ["#d62728", "#2ca02c", "#1f77b4"])):
        ax.bar(x + (i - 1) * 0.27, [medias[c][i] for c in CLASSES], 0.27, label=canal, color=cor)
    ax.set_xticks(x, CLASSES)
    ax.set_ylabel("Intensidade média (0–255)")
    ax.set_title("Cor média por classe")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset-resized")
    ap.add_argument("--out_json", default="split.json")
    ap.add_argument("--fig_dir", default="figuras/eda")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Listagem e integridade
    paths, labels = listar_imagens(data_dir)
    corrompidas, tamanhos, modos, duplicatas = checar_integridade(paths)
    print(f"Total de imagens: {len(paths)} (esperado: 2527)")
    print(f"Corrompidas: {len(corrompidas)}")
    for p, e in corrompidas:
        print("   ", p, e)
    print(f"Tamanhos encontrados: {dict(tamanhos)}")
    if set(tamanhos) != {EXPECTED_SIZE}:
        print(f"  ATENÇÃO: há imagens fora de {EXPECTED_SIZE}")
    print(f"Modos de cor (canais): {dict(modos)}  -> todas são convertidas para RGB no carregamento")
    print(f"Grupos de duplicatas exatas: {len(duplicatas)}")
    for g in duplicatas:
        print("   ", g)

    # remove corrompidas antes do split
    ruins = {p for p, _ in corrompidas}
    pares = [(str(p), l) for p, l in zip(paths, labels) if str(p) not in ruins]
    paths_ok = [p for p, _ in pares]
    labels_ok = [l for _, l in pares]

    # 2. Contagem
    contagem = Counter(labels_ok)
    print("\nContagem por classe:")
    for c in CLASSES:
        print(f"  {c:10s} {contagem[c]:5d}  ({contagem[c] / len(labels_ok):.1%})")
    print(f"  Razão maior/menor classe: {max(contagem.values()) / min(contagem.values()):.2f}")

    # 3. Split estratificado 70/15/15, seed 42
    # Tratamento das duplicatas: todas as cópias de uma mesma imagem vão para o TREINO.
    # Se uma cópia caísse no treino e outra na validação/teste, o modelo seria avaliado
    # numa imagem que já viu (vazamento). Como as cópias têm rótulos diferentes, ficam
    # registradas como ruído de rótulo (limitação do dataset).
    em_duplicata = {p for g in duplicatas for p in g}
    fixos = [(p, l) for p, l in pares if p in em_duplicata]
    livres = [(p, l) for p, l in pares if p not in em_duplicata]
    print(f"\nImagens em duplicatas forçadas no treino: {len(fixos)}")

    # proporção do treino ajustada para o total continuar ~70/15/15
    n_total = len(pares)
    n_val_test = round(0.30 * n_total)
    X_livre = [p for p, _ in livres]
    y_livre = [l for _, l in livres]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X_livre, y_livre, test_size=n_val_test, stratify=y_livre, random_state=SEED)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
    X_tr = X_tr + [p for p, _ in fixos]
    y_tr = y_tr + [l for _, l in fixos]

    split_counts = {
        "train": Counter(y_tr), "val": Counter(y_va), "test": Counter(y_te)}
    print(f"Tamanhos: train={len(X_tr)} val={len(X_va)} test={len(X_te)}")

    # sem vazamento entre conjuntos: nem por caminho, nem por conteúdo (hash)
    assert not (set(X_tr) & set(X_va) or set(X_tr) & set(X_te) or set(X_va) & set(X_te))
    hash_de = {p: hashlib.md5(Path(p).read_bytes()).hexdigest() for p in paths_ok}
    h_tr, h_va, h_te = ({hash_de[p] for p in X} for X in (X_tr, X_va, X_te))
    assert not (h_tr & h_va or h_tr & h_te or h_va & h_te), "Mesma imagem em dois conjuntos!"
    print("Checagem de vazamento por conteúdo: OK (nenhuma imagem repetida entre conjuntos)")

    # 4. Pesos de classe — SÓ com o treino
    pesos = compute_class_weight("balanced", classes=np.array(CLASSES), y=np.array(y_tr))
    pesos = {c: round(float(w), 4) for c, w in zip(CLASSES, pesos)}
    print(f"Pesos de classe (treino): {pesos}")

    # caminhos relativos à pasta do dataset (funciona em qualquer máquina/Colab)
    rel = lambda ps: [str(Path(p).relative_to(data_dir)) for p in ps]
    split = {
        "seed": SEED,
        "classes": CLASSES,
        "data_dir": data_dir.name,
        "class_weights": pesos,
        "counts": {k: {c: v[c] for c in CLASSES} for k, v in split_counts.items()},
        "duplicatas_no_treino": [rel(g) for g in duplicatas],
        "train": [{"path": p, "label": l} for p, l in zip(rel(X_tr), y_tr)],
        "val":   [{"path": p, "label": l} for p, l in zip(rel(X_va), y_va)],
        "test":  [{"path": p, "label": l} for p, l in zip(rel(X_te), y_te)],
    }
    Path(args.out_json).write_text(json.dumps(split, indent=2, ensure_ascii=False))
    print(f"\nSalvo: {args.out_json}")

    # 5. Figuras
    fig_distribuicao(contagem, fig_dir / "distribuicao_classes.png")
    fig_distribuicao_split(split_counts, fig_dir / "distribuicao_split.png")
    fig_amostras(paths_ok, labels_ok, fig_dir / "amostras_por_classe.png")
    fig_intensidade(paths_ok, labels_ok, fig_dir / "cor_media_por_classe.png")
    fig_resolucoes(X_te[0], fig_dir / "mesma_imagem_tres_resolucoes.png")
    fig_duplicatas(duplicatas, data_dir, fig_dir / "duplicatas.png")
    print(f"Figuras salvas em: {fig_dir}/")


if __name__ == "__main__":
    main()
