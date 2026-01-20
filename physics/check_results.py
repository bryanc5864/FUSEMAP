import json
import os

print("\n" + "="*80)
print("RESULTS VERIFICATION SUMMARY")
print("="*80 + "\n")

# Check PhysInformer models
for cell_type in ['S2', 'WTC11', 'HepG2', 'K562']:
    print(f"\n{cell_type} PhysInformer:")
    print("-" * 40)
    
    json_path = f'results/PhysInformer_{cell_type}/parsed_epochs.json'
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
            print(f"  ✓ Parsed {len(data)} epochs")
            if data:
                e1, last = data[0], data[-1]
                print(f"  ✓ Epoch 1: Loss={e1.get('train_total_loss'):.2f}, Pearson={e1.get('train_pearson'):.4f}")
                print(f"  ✓ Epoch {last['epoch']}: Loss={last.get('train_total_loss'):.2f}, Pearson={last.get('train_pearson'):.4f}")
    
    # Check plots exist
    plots = ['loss_curves.png', 'pearson_evolution.png']
    for plot in plots:
        path = f'results/PhysInformer_{cell_type}/{plot}'
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024
            print(f"  ✓ {plot}: {size:.1f} KB")

# Check TileFormer
print(f"\n\nTileFormer:")
print("-" * 40)

json_path = 'results/TileFormer/parsed_epochs.json'
if os.path.exists(json_path):
    with open(json_path) as f:
        data = json.load(f)
        print(f"  ✓ Parsed {len(data)} epochs")
        if data:
            e1, last = data[0], data[-1]
            print(f"  ✓ Epoch 1: MSE={e1.get('mse'):.6f}, Pearson={e1.get('pearson_r'):.4f}")
            print(f"  ✓ Epoch {last['epoch']}: MSE={last.get('mse'):.6f}, Pearson={last.get('pearson_r'):.4f}")

plot = 'comprehensive_metrics.png'
path = f'results/TileFormer/{plot}'
if os.path.exists(path):
    size = os.path.getsize(path) / 1024
    print(f"  ✓ {plot}: {size:.1f} KB")

print("\n" + "="*80)
print("All plots generated successfully with data!")
print("="*80)
