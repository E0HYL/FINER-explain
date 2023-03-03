X: padded features of shape (num_samples, 20000, 18) to feed VGG
- num_samples = 1 for feature files in `x_malware_gt`

y: shape (num_samples, 2)
- [0, 1] for malware and [1, 0] for goodware