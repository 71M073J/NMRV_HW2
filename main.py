import numpy as np
import matplotlib.pyplot as plt
from ex2_utils import generate_responses_1, Tracker, extract_histogram, backproject_histogram,\
    create_epanechnik_kernel, get_patch
from ncc_tracker_example import NCCTracker, NCCParams
from sequence_utils import VOTSequence

class MeanShiftTrack(Tracker):
    def initialize(self, image, region):
        minx, miny = region[0], region[1]
        maxx, maxy = region[0] + region[2], region[1] + region[3]
        extracted = image[:, minx:maxx, miny:maxy].transpose((1,2,0))
        print(extracted.shape)
        self.bins = 5
        sigma = 1
        self.alpha = 0.1
        self.eps = 1e-8
        self.hist_kernel = create_epanechnik_kernel(maxx - minx, maxy - miny, sigma)
        self.der_kernel = (self.hist_kernel > 0).astype(np.int32)
        print(self.hist_kernel.shape)
        self.minx, self.miny, self.maxx, self.maxy = minx, miny, maxx, maxy
        #calc_Weights = np.sqrt(q(extracted)/p(extracted))
        self.template_hist = extract_histogram(extracted, self.bins, weights=self.hist_kernel)
        #projection = backproject_histogram(image, self.template_hist)
        print(self.template_hist.shape)
        ...
    def track(self, image):

        extracted = image[:, self.minx:self.maxx, self.miny:self.maxy].transpose((1,2,0))

        next_hist = extract_histogram(extracted, self.bins, weights=self.hist_kernel)
        weights_v = np.sqrt(self.template_hist/(next_hist + self.eps))

        weights_wi = backproject_histogram(extracted, weights_v, self.bins)
        x, y = np.meshgrid(self.template_hist, next_hist)
        print(x[0], y[:, 0], x[0].shape, x[1].shape)
        print(weights_wi * x)
        quit()
        plt.imshow(weights_wi)
        plt.show()
        print(weights_wi, weights_wi.shape)
        ...
        quit()
        new_localised_hist = ...
        self.template_hist = (1 - self.alpha) * self.template_hist + self.alpha * new_localised_hist





if __name__ == '__main__':
    a = np.random.randint(0,255, size=(3, 100, 100))
    #plt.imshow(a.transpose((1,2,0)))
    #plt.show()
    T = MeanShiftTrack("")
    T.initialize(a, [40, 60, 21, 21])
    T.track(np.roll(a, 3, axis=-1))
    quit()
    a = generate_responses_1()
    tr = NCCTracker(NCCParams())
    ins = [40,60,21,21]
    tr.initialize(a, ins)
    nrol = 7
    out = tr.track(np.roll(a, nrol))
    print(out)
    fig, ax = plt.subplots(1,2)
    print(a, a.shape)
    ax[0].imshow(a)
    ax[0].plot([ins[0], ins[0], ins[0] + ins[2], ins[0] + ins[2], ins[0]],
               [ins[1], ins[1] + ins[3], ins[1] + ins[3], ins[1], ins[1]], color="r")
    ax[1].imshow(np.roll(a, nrol))
    ax[1].plot([out[0], out[0], out[0] + out[2], out[0] + out[2], out[0]],
               [out[1], out[1] + out[3], out[1] + out[3], out[1], out[1]], color="r")
    plt.show()