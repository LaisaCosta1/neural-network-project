"""
02_mlp.py — Experimentos MLP (integrante 2)

Treina:
  mlp_128_sem   MLP, 128×96, sem pesos de classe
  mlp_128_com   MLP, 128×96, com pesos de classe (calculados só no treino, vindos do split.json)

Usa o common.py (mesmo split, semente 42, mesmo pipeline e mesmos callbacks dos outros modelos).
O conjunto de TESTE não é usado aqui — só no 03_resultados.py, no final.

Uso (na raiz do projeto, com a pasta dataset-resized/ ao lado do split.json):
  python 02_mlp.py                      # treina os dois (pula o que já terminou)
  python 02_mlp.py --exp mlp_128_sem    # só um
  python 02_mlp.py --forcar             # treina de novo do zero

Arquitetura (justificativa resumida, para o artigo):
  - Entrada: imagem 96×128×3 achatada -> vetor de 36.864 valores. O MLP não tem noção de
    vizinhança entre pixels; ele serve de linha de base para mostrar o ganho da CNN.
  - 2 camadas ocultas (256 e 128 neurônios, ReLU). A primeira camada já tem ~9,4 milhões de
    pesos; com só 1.768 imagens de treino, camadas maiores decoram o treino (overfitting).
  - BatchNormalization estabiliza o treino com entradas de dimensão alta.
  - Dropout (0,5 e 0,3) e regularização L2 (1e-4) contra overfitting.
  - Adam com lr 3e-4, ReduceLROnPlateau e EarlyStopping em val_loss (paciência 10),
    guardando os pesos da melhor época na validação.
  Os valores podem ser ajustados pelas opções abaixo, SEMPRE olhando só a validação
  (F1 macro de validação é a métrica de seleção pedida no enunciado). Se fizer busca de
  hiperparâmetros, registre cada configuração testada numa tabela separada.
  mlp_128_com usa exatamente a mesma configuração e semente do mlp_128_sem; só muda a perda,
  que passa a ser ponderada pelos pesos de classe (inversamente proporcionais à frequência no treino).
"""
import argparse

from common import treinar_experimento

EXPERIMENTOS_MLP = ["mlp_128_sem", "mlp_128_com"]


def construir_mlp(input_shape, num_classes, unidades=(256, 128), dropouts=(0.5, 0.3), l2=1e-4):
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers

    entrada = tf.keras.Input(shape=input_shape)
    x = layers.Flatten()(entrada)
    for n, d in zip(unidades, dropouts):
        x = layers.Dense(n, kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(d)(x)
    saida = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(entrada, saida, name="mlp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=EXPERIMENTOS_MLP, nargs="+", default=EXPERIMENTOS_MLP)
    ap.add_argument("--split", default="split.json")
    ap.add_argument("--data_dir", default="dataset-resized")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--unidades", type=int, nargs="+", default=[256, 128])
    ap.add_argument("--dropouts", type=float, nargs="+", default=[0.5, 0.3])
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--epocas", type=int, default=None, help="padrão: o do common.py (100)")
    ap.add_argument("--forcar", action="store_true", help="treina de novo mesmo se já existir")
    args = ap.parse_args()

    if len(args.unidades) != len(args.dropouts):
        ap.error("--unidades e --dropouts precisam ter o mesmo tamanho")

    def construir(input_shape, num_classes):
        return construir_mlp(input_shape, num_classes, args.unidades, args.dropouts, args.l2)

    resumo = {}
    for nome in args.exp:
        print("\n" + "=" * 70 + f"\n{nome}\n" + "=" * 70)
        meta = treinar_experimento(
            nome, construir, lr=args.lr, split_path=args.split, data_dir=args.data_dir,
            epocas_max=args.epocas, forcar=args.forcar,
            info_extra={"arquitetura": {"unidades": args.unidades, "dropouts": args.dropouts,
                                        "l2": args.l2, "batchnorm": True, "ativacao": "relu"}})
        if meta:
            resumo[nome] = meta

    if resumo:
        print("\nResumo (VALIDAÇÃO — o teste fica para o 03_resultados.py):")
        for nome, m in resumo.items():
            print(f"  {nome:12s} val_F1_macro={m['val_f1_macro']:.3f}  val_acc={m['val_accuracy']:.3f}  "
                  f"época salva={m['epoca_do_modelo_salvo']}/{m['epocas_rodadas']}  "
                  f"tempo={m['tempo_total_treino_s'] / 60:.1f} min ({m['tempo_medio_por_epoca_s']:.1f} s/época)  "
                  f"params={m['parametros_treinaveis']:,}  {m['tamanho_modelo_salvo_mb']} MB")


if __name__ == "__main__":
    main()
