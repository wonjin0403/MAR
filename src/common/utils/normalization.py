def min_max_normalization(img, min_v=0, max_v=4095, min_new=0.0, max_new=1.0):
    range_v = max_new - min_new
    return (img-min_v)/(max_v-min_v) * range_v + min_new

def min_max2original(img, min_v=0, max_v=4095, min_new=0.0, max_new=1.0):
    range_v = max_new - min_new
    return (img - min_new) / range_v * (max_v-min_v) + min_v