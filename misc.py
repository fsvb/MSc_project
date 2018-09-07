def super_convergence(lr_range=(0.001, 0.01), momentum_range=(0.95, 0.85), epochs_for_cycle=9):
    def triangle(e, min, max):
        # Linearly increases quantity from min to max, and back to min again in `epochs_for_cycle`-many
        # iterations. After that, stays at min.
        peak_at = epochs_for_cycle // 2
        slope = (max - min) / peak_at
        if e <= peak_at:
            return min + slope * e
        elif peak_at < e <= 2 * peak_at:
            return max - slope * (e - peak_at)
        else:
            return min

    lr = lambda e: triangle(e, lr_range[0], lr_range[1])
    mom = lambda e: triangle(e, momentum_range[0], momentum_range[1])  # triangle works symmetrically
    return lr, mom
