from sklearn.metrics import roc_curve

# Inside your validation loop:
all_scores = []
all_labels = []

with torch.no_grad():
    for waveforms, labels in val_loader:
        waveforms = waveforms.to(device)
        outputs = model(waveforms)
        all_scores.extend(torch.sigmoid(outputs).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate EER
fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
fnr = 1 - tpr
eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]  # EER point

print(f"\nEqual Error Rate (EER): {eer*100:.2f}%")