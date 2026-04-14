# Download and load the CIFAR-100 dataset (100 fine-grained classes).
# Apply standard training augmentation: random horizontal flip, random crop,
#   and optionally AutoAugment or RandAugment for improved generalisation.
# Apply standard normalisation: mean=(0.5071, 0.4867, 0.4408),
#                                std =(0.2675, 0.2565, 0.2761).
# Return separate DataLoaders for train and test splits.
# Used in all experiments reported in the paper (ResNet-18 backbone).
