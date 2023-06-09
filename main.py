import cv2
import numpy as np
import matplotlib.pyplot as plt
from ex2_utils import generate_responses_1, Tracker, extract_histogram, backproject_histogram, \
    create_epanechnik_kernel, get_patch
from ncc_tracker_example import NCCTracker, NCCParams
from sequence_utils import VOTSequence


def extract_hist(extracted, bins, weights):
    x = extract_histogram(extracted, bins, weights=weights)
    return x / np.sum(x)


def mode_tracking(image, center, offsets, kernel, niter=1000, eps=1e-4, factor=1.0, ax=None):
    minx, maxx = center[0] - offsets[0], center[0] + offsets[0] + 1
    miny, maxy = center[1] - offsets[1], center[1] + offsets[1] + 1
    sizex, sizey = (abs(maxx - minx) + 0, abs(maxy - miny) + 0)
    extracted, _ = get_patch(image, ((minx + maxx) / 2, (miny + maxy) / 2), (sizex, sizey))
    curx, cury = center
    print(curx, cury)
    # kernel = kernel / np.sum(kernel)
    # kernel = (kernel > 0).astype(int)
    xs = [curx]
    ys = [cury]
    for i in range(niter):
        extracted, _ = get_patch(image, [curx, cury], [sizex, sizey])  # TO JE wi v enačbi
        # print(extracted.shape)

        x, y = np.meshgrid(np.arange(-(sizex // 2), sizex // 2 + 1, 1),  # TO JE xi v enačbi
                           np.arange(-(sizey // 2), sizey // 2 + 1, 1))
        # print(x, y)
        g_w = extracted * kernel
        sumgw = np.sum(g_w)
        dx = np.sum(x * g_w) / sumgw
        dy = np.sum(y * g_w) / sumgw

        # print(dx, dy)
        if abs(dx) + abs(dy) < eps:
            print("WAT ZE FUG")
            break
        curx += dx * factor
        cury += dy * factor
        xs.append(curx)
        ys.append(cury)
        # plt.imshow(image)
        # plt.show()
    print(curx, cury)
    if ax is None:
        plt.imshow(image)
        plt.plot(xs, ys, color="r")
        plt.show()
    else:
        ax.imshow(image[30:90, 30:90])
        ax.plot(np.array(xs) - 30, np.array(ys) - 30, color="r")
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

class MSParams:
    def __init__(self, bins=10, sigma=0.3, niter=10, alpha=0.1, eps=1e-8, colorspace=False):
        self.bins = bins
        self.sigma = sigma
        self.niter = niter
        self.alpha = alpha
        self.eps = eps
        self.colorspace = colorspace


class MeanShiftTracker(Tracker):
    def initialize(self, image, region):

        self.colorspace = self.parameters.colorspace
        if self.colorspace:
            # TODO SWITCH?
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        minx, miny = region[0], region[1]
        maxx, maxy = region[0] + region[2], region[1] + region[3]
        # print(minx, maxx, miny, maxy)
        # extracted = image[minx:maxx, miny:maxy, :]
        extracted, _ = get_patch(image, ((minx + maxx) / 2, (miny + maxy) / 2), (maxx - minx, maxy - miny))
        # print(extracted.shape)
        # TODO nastavi vse te vrednosti iz self.params
        # Nah
        self.bins = self.parameters.bins
        self.sigma = self.parameters.sigma
        self.niter = self.parameters.niter
        self.alpha = self.parameters.alpha
        self.eps = self.parameters.eps
        self.hist_kernel = create_epanechnik_kernel(round(maxx - minx), round(maxy - miny),
                                                    self.sigma)  # [:, :, np.newaxis]
        self.reduced_dim0 = self.hist_kernel.shape[0] == extracted.shape[0]
        self.reduced_dim1 = self.hist_kernel.shape[1] == extracted.shape[1]
        self.hist_kernel = self.hist_kernel[:extracted.shape[0], :extracted.shape[1]]
        self.sizex = self.hist_kernel.shape[1]
        self.sizey = self.hist_kernel.shape[0]
        self.der_kernel = (self.hist_kernel > 0).astype(np.int32)
        # print(self.hist_kernel.shape)
        self.minx, self.miny, self.maxx, self.maxy = minx, miny, maxx, maxy

        # calc_Weights = np.sqrt(q(extracted)/p(extracted))
        self.template_hist = extract_hist(extracted, self.bins, self.hist_kernel)
        # projection = backproject_histogram(image, self.template_hist)
        # print(self.template_hist.shape)
        ...

    def track(self, image):
        if self.colorspace:
            # TODO SWITCH?
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # extracted = image[self.minx:self.maxx, self.miny:self.maxy, :]
        next_hist = 0
        plt.imshow(image)
        for i in range(self.niter):
            extracted, _ = get_patch(image, ((self.minx + self.maxx) / 2, (self.miny + self.maxy) / 2), (
                self.sizex, self.sizey))
            next_hist = extract_hist(extracted, self.bins, self.hist_kernel)  # to je p v enačbi

            weights_v = np.sqrt(self.template_hist / (next_hist + self.eps))

            weights_wi = backproject_histogram(extracted, weights_v, self.bins)  # wi v enačbi
            x, y = np.meshgrid(np.arange(-(self.sizex // 2), self.sizex // 2 + int(self.reduced_dim1), 1),
                               # TO JE xi v enačbi
                               np.arange(-(self.sizey // 2), self.sizey // 2 + int(self.reduced_dim0), 1))

            g_w = weights_wi * self.der_kernel

            sumgw = np.sum(g_w)
            dx = np.sum(x * g_w) / sumgw
            dy = np.sum(y * g_w) / sumgw
            self.minx += dx
            self.maxx += dx
            self.miny += dy
            self.maxy += dy

        # extracted, _ = get_patch(image, ((self.minx + self.maxx) / 2, (self.miny + self.maxy) / 2), (
        #    self.sizex, self.sizey))
        # next_hist = extract_hist(extracted, self.bins, self.hist_kernel)  # to je p v enačbi
        self.template_hist = (1 - self.alpha) * self.template_hist + self.alpha * next_hist
        self.template_hist /= np.sum(self.template_hist)
        return [self.minx, self.miny, self.sizex, self.sizey]
    # print(weights_wi * x)
    # quit()
    # plt.imshow(weights_wi)
    # plt.show()
    # print(weights_wi, weights_wi.shape)
    # ...
    # quit()
    # new_localised_hist = ...
    # self.template_hist = (1 - self.alpha) * self.template_hist + self.alpha * new_localised_hist


if __name__ == '__main__':
    a = np.random.randint(0, 255, size=(100, 100, 3))
    a = generate_responses_1()
    a = np.arange(0,100,1)
    a = np.sin(a[:,np.newaxis]/10)
    print(a)
    a = a.dot(a.T)
    a *= np.arange(0,100,1)
    a -= np.min(a)
    a /= np.sum(a)



    # a = np.arange(0,100,1)[:,np.newaxis].dot(np.arange(100, 0,-1)[np.newaxis, :])
    fig, ax = plt.subplots(4, 4)
    for id2, j in enumerate([2, 5, 10, 15]):
        for id, i in enumerate([30, 50, 70, 80]):
            mode_tracking(a, (i, 60), (j, j), create_epanechnik_kernel(2 * j + 1, 2 * j + 1, 1), niter=1000,
                          ax=ax[id2, id], factor=1)
    plt.tight_layout()
    plt.show()
    quit()
    # plt.imshow(a.transpose((1,2,0)))
    # plt.show()
    T = MeanShiftTrack("")
    T.initialize(a, [40, 60, 21, 21])
    T.track(np.roll(a, 3, axis=-1))
    quit()
    a = generate_responses_1()
    tr = NCCTracker(NCCParams())
    ins = [40, 60, 21, 21]
    tr.initialize(a, ins)
    nrol = 7
    out = tr.track(np.roll(a, nrol))
    print(out)
    fig, ax = plt.subplots(1, 2)
    print(a, a.shape)
    ax[0].imshow(a)
    ax[0].plot([ins[0], ins[0], ins[0] + ins[2], ins[0] + ins[2], ins[0]],
               [ins[1], ins[1] + ins[3], ins[1] + ins[3], ins[1], ins[1]], color="r")
    ax[1].imshow(np.roll(a, nrol))
    ax[1].plot([out[0], out[0], out[0] + out[2], out[0] + out[2], out[0]],
               [out[1], out[1] + out[3], out[1] + out[3], out[1], out[1]], color="r")
    plt.show()
