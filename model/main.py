import numpy as np

from model.imvs.engine.imvs import IMVS


def main():
    # Initialize model
    imvs = IMVS(use_cuda=True)

    # -----------------------------
    # 1. Single random image
    # -----------------------------
    # Example shape: (H, W, C) = (256, 256, 3)
    img1 = np.random.randint(
        low=0,
        high=3001,  # upper bound is exclusive
        size=(256, 256, 3),
        dtype=np.uint16
    )

    # Pass to model (adjust method name if needed)
    output1 = imvs.infer(img1, None)
    print("Output 1 shape:", output1.shape)


    # -----------------------------
    # 2. Random image + scribble mask
    # -----------------------------
    img2 = np.random.randint(
        low=0,
        high=3001,
        size=(256, 256, 3),
        dtype=np.uint16
    )

    # Boolean scribble mask (same spatial size, usually no channels)
    scribble_mask = np.random.choice(
        a=[False, True],
        size=(256, 256)
    )

    # Pass both to model
    output2 = imvs.infer(img2, scribble_mask)
    print("Output 2 shape:", output2.shape)


if __name__ == "__main__":
    main()
