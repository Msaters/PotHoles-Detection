import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def plot_comprehensive_logs(csv_path: Path, output_dir: Path):
    if not csv_path.exists():
        print(f"❌ Nie znaleziono pliku: {csv_path}")
        return

    # Wczytanie danych
    df = pd.read_csv(csv_path)

    # PyTorch Lightning zapisuje logi treningowe i walidacyjne w różnych wierszach.
    # Najlepszym sposobem na czysty wykres jest pogrupowanie ich po numerze epoki (uśrednienie).
    if 'epoch' not in df.columns:
        print("❌ Plik CSV nie zawiera kolumny 'epoch'.")
        return

    epoch_df = df.groupby('epoch').mean().reset_index()

    # Definiujemy, jakich kolumn szukamy (dla treningu i walidacji)
    metrics_to_plot = [
        {"title": "Total Loss (Główny błąd)", "train": "train/loss", "val": "val/loss"},
        {"title": "Box Loss (Dokładność ramek)", "train": "train/box_loss", "val": "val/box_loss"},
        {"title": "Class Loss (Rozpoznawanie klasy)", "train": "train/cls_loss", "val": "val/cls_loss"},
        {"title": "DFL Loss (Precyzja krawędzi)", "train": "train/dfl_loss", "val": "val/dfl_loss"}
    ]

    # Ustawienia płótna (2 rzędy, 2 kolumny)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Szczegółowy Raport Treningu YOLO', fontsize=18, fontweight='bold', y=1.02)

    # Rysowanie każdego z 4 wykresów
    for idx, metric in enumerate(metrics_to_plot):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        train_col = metric["train"]
        val_col = metric["val"]

        # Rysuj linię treningową (jeśli istnieje)
        if train_col in epoch_df.columns:
            ax.plot(epoch_df['epoch'], epoch_df[train_col], 
                    label='Trening', color='#2196f3', linewidth=2, marker='o', markersize=4)
        
        # Rysuj linię walidacyjną (jeśli istnieje)
        if val_col in epoch_df.columns:
            ax.plot(epoch_df['epoch'], epoch_df[val_col], 
                    label='Walidacja', color='#f44336', linewidth=2, marker='s', markersize=4)

        ax.set_title(metric["title"], fontsize=14)
        ax.set_xlabel('Epoka', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    
    # Zapis
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "training_curves_dashboard.png"
    plt.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Zapisano szczegółowy panel wykresów w: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generuj wykresy po treningu modelu.")
    parser.add_argument('--run_dir', type=Path, required=True, 
                        help='Ścieżka do folderu z danym eksperymentem (tam gdzie jest metrics.csv)')
    parser.add_argument('--output', type=Path, default=Path('outputs/yolo_report'), 
                        help='Gdzie zapisać wykresy')
    
    args = parser.parse_args()
    
    # Szukamy pliku metrics.csv w podanym folderze
    csv_file = args.run_dir / "metrics.csv"
    
    plot_comprehensive_logs(csv_file, args.output)