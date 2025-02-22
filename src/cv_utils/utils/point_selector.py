import matplotlib.pyplot as plt
import numpy as np


class DoubleImagesPointSelector:
    """
    A class to display two images side by side and select corresponding points on both images.

    Attributes:
        left_image (np.ndarray): The left image to display.
        right_image (np.ndarray): The right image to display.
        num_points (int): The number of points to select.
        points (List[Tuple[float, float, float, float]]): List to store selected points as (left_x, left_y, right_x, right_y).
        fig (matplotlib.figure.Figure): The matplotlib figure object.
        ax1 (matplotlib.axes._subplots.AxesSubplot): Axis for the left image.
        ax2 (matplotlib.axes._subplots.AxesSubplot): Axis for the right image.
        ax1_zoom (matplotlib.axes._subplots.AxesSubplot): Axis for the zoomed view of the left image.
        ax2_zoom (matplotlib.axes._subplots.AxesSubplot): Axis for the zoomed view of the right image.
        point_count (int): Counter for the number of points selected.
        selected_left (Tuple[float, float]): Coordinates of the selected point in the left image.
        current_image (str): Indicates the current image being processed ('Left' or 'Right').
        zoom_size (int): Size of the zoom window around the selected point.
        zoom_image_coords (Tuple[int, int, float, float]): Coordinates of the zoomed area.
        original_point (Tuple[float, float]): Original coordinates of the selected point in the zoom window.
    """

    def __init__(self, left_image, right_image, num_points):
        self.left_image = left_image
        self.right_image = right_image
        self.num_points = num_points
        self.points = []  # To store (left_x, left_y, right_x, right_y)
        self.fig, ((self.ax1, self.ax2), (self.ax1_zoom, self.ax2_zoom)) = plt.subplots(
            2, 2, figsize=(12, 12)
        )
        self.point_count = 0
        self.selected_left = None
        self.current_image = None
        self.zoom_size = 40  # Size of the zoom window around the selected point
        self.zoom_image_coords = None  # To store the coordinates of the zoomed area
        self.original_point = (
            None  # To store the original selected point coordinates for zoom window
        )

        self._init_zoom()
        self.connect()

    def connect(self):
        """Connect to all the needed events."""
        self.ax1.imshow(self.left_image)
        self.ax1.set_title(f"Left Image: Select Point 0 of {self.num_points}")
        self.ax1.axis("off")

        self.ax2.imshow(self.right_image)
        self.ax2.set_title("Right Image: Waiting for Left")
        self.ax2.axis("off")

        self.cid_click = self.fig.canvas.mpl_connect(
            "button_press_event", self._onclick
        )
        self.cid_key = self.fig.canvas.mpl_connect("key_press_event", self._onkey)
        plt.show()

    def _init_zoom(self):
        """Initialize the zoom area with a placeholder."""
        placeholder = np.full(
            (self.zoom_size * 2, self.zoom_size * 2, 3), 128, dtype=np.uint8
        )  # Gray placeholder
        self.ax1_zoom.imshow(placeholder)
        self.ax1_zoom.set_title("Left Zoom Window")
        self.ax1_zoom.axis("off")

        self.ax2_zoom.imshow(placeholder)
        self.ax2_zoom.set_title("Right Zoom Window")
        self.ax2_zoom.axis("off")

    def _onclick(self, event):
        if event.inaxes in [self.ax1, self.ax2]:
            if (
                event.inaxes == self.ax1
                and self.point_count < self.num_points
                and self.selected_left is None
            ):
                self._update_zoom_image(
                    event.xdata, event.ydata, self.left_image, "Left"
                )
            elif event.inaxes == self.ax2 and self.selected_left is not None:
                self._update_zoom_image(
                    event.xdata, event.ydata, self.right_image, "Right"
                )
        elif event.inaxes in [self.ax1_zoom, self.ax2_zoom]:
            self._adjust_point_in_zoom(event)

    def _update_zoom_image(self, x, y, image, image_side):
        x0, x1 = int(max(0, x - self.zoom_size)), int(
            min(image.shape[1], x + self.zoom_size)
        )
        y0, y1 = int(max(0, y - self.zoom_size)), int(
            min(image.shape[0], y + self.zoom_size)
        )
        zoom_img = image[y0:y1, x0:x1]
        self.zoom_image_coords = (x0, y0, x, y)
        self.current_image = image_side

        ax_zoom = self.ax1_zoom if image_side == "Left" else self.ax2_zoom
        ax_zoom.clear()
        ax_zoom.imshow(zoom_img)
        ax_zoom.plot(x - x0, y - y0, "ro", markersize=10)  # Red: initial selected point
        self.original_point = (x - x0, y - y0)
        ax_zoom.set_title("Zoom View - Click to Adjust")
        ax_zoom.axis("off")
        self.fig.canvas.draw()

    def _adjust_point_in_zoom(self, event):
        if self.zoom_image_coords:
            x0, y0, _, _ = self.zoom_image_coords
            new_x, new_y = x0 + event.xdata, y0 + event.ydata
            ax_zoom = self.ax1_zoom if self.current_image == "Left" else self.ax2_zoom
            ax_zoom.plot(
                event.xdata, event.ydata, "bo", markersize=10
            )  # Blue: adjusted point
            ax = self.ax1 if self.current_image == "Left" else self.ax2
            ax.plot(
                new_x, new_y, "bo", markersize=5
            )  # Mark the adjusted point on the original image
            if self.current_image == "Left":
                self.selected_left = (new_x, new_y)
                ax.set_title("Left Image: Waiting for Right Selection")
            elif self.current_image == "Right":
                self.points.append((*self.selected_left, new_x, new_y))
                self.point_count += 1
                self.selected_left = None
                if self.point_count < self.num_points:
                    self.ax1.set_title(
                        f"Left Image: Select Point {self.point_count} of {self.num_points}"
                    )
                else:
                    self.ax1.set_title(
                        'Selection Completed - Press "r" to reset or "q" to quit'
                    )
                    self.ax2.set_title(
                        'Selection Completed - Press "r" to reset or "q" to quit'
                    )
            self.fig.canvas.draw()

    def _onkey(self, event):
        if event.key == "r":
            self._reset()
        elif event.key == "q":
            plt.close(self.fig)

    def _reset(self):
        """Reset the point selection."""
        self.points = []
        self.point_count = 0
        self.selected_left = None
        self.current_image = None
        self.ax1.clear()
        self.ax2.clear()
        self.ax1_zoom.clear()
        self.ax2_zoom.clear()
        self.ax1.imshow(self.left_image)
        self.ax1.set_title(f"Left Image: Select Point 0 of {self.num_points}")
        self.ax1.axis("off")
        self.ax2.imshow(self.right_image)
        self.ax2.set_title("Right Image: Waiting for Left")
        self.ax2.axis("off")
        self._init_zoom()
        self.fig.canvas.draw()

    def reset(self):
        self._reset()

    def get_points(self):
        pts1 = [(x1, y1) for (x1, y1, _, _) in self.points]
        pts2 = [(x2, y2) for (_, _, x2, y2) in self.points]
        return pts1, pts2


class SingleImagePointSelector:
    """
    A class to display a single image and select points on it with a zoomed-in view for fine adjustments.

    Attributes:
        image (np.ndarray): The image to display and select points on.
        num_points (int): The number of points to select.
        points (List[Tuple[float, float]]): List to store selected points as (x, y) coordinates.
        fig (matplotlib.figure.Figure): The matplotlib figure object.
        ax (matplotlib.axes._subplots.AxesSubplot): Axis for the main image.
        ax_zoom (matplotlib.axes._subplots.AxesSubplot): Axis for the zoomed view.
        point_count (int): Counter for the number of points selected.
        zoom_size (int): Size of the zoom window around the selected point.
        zoom_image_coords (Tuple[int, int, float, float]): Coordinates of the zoomed area.
    """

    def __init__(self, image, num_points):
        self.image = image
        self.num_points = num_points
        self.points = []  # To store (x, y) coordinates
        self.fig, (self.ax, self.ax_zoom) = plt.subplots(1, 2, figsize=(12, 6))
        self.point_count = 0
        self.zoom_size = 40  # Size of the zoom window around the selected point
        self.zoom_image_coords = None  # To store the coordinates of the zoomed area

        self._init_zoom()
        self.connect()

    def connect(self):
        """Connect to all the needed events."""
        self.ax.imshow(self.image)
        self.ax.set_title("Image: Select Point 0 of {self.num_points}")
        self.ax.axis("off")

        self.cid_click = self.fig.canvas.mpl_connect(
            "button_press_event", self._onclick
        )
        self.cid_key = self.fig.canvas.mpl_connect("key_press_event", self._onkey)
        plt.show()

    def _init_zoom(self):
        """Initialize the zoom area with a placeholder."""
        placeholder = np.full(
            (self.zoom_size * 2, self.zoom_size * 2, 3), 128, dtype=np.uint8
        )  # Gray placeholder
        self.ax_zoom.imshow(placeholder)
        self.ax_zoom.set_title("Zoom Window")
        self.ax_zoom.axis("off")

    def _onclick(self, event):
        if event.inaxes == self.ax:
            self._update_zoom_image(event.xdata, event.ydata)
        elif event.inaxes == self.ax_zoom:
            self._adjust_point_in_zoom(event)

    def _update_zoom_image(self, x, y):
        x0, x1 = int(max(0, x - self.zoom_size)), int(
            min(self.image.shape[1], x + self.zoom_size)
        )
        y0, y1 = int(max(0, y - self.zoom_size)), int(
            min(self.image.shape[0], y + self.zoom_size)
        )
        zoom_img = self.image[y0:y1, x0:x1]
        self.zoom_image_coords = (x0, y0, x, y)

        self.ax_zoom.clear()
        self.ax_zoom.imshow(zoom_img)
        self.ax_zoom.plot(
            x - x0, y - y0, "ro", markersize=10
        )  # Mark the selected point
        self.ax_zoom.set_title("Zoom View - Click to Adjust")
        self.ax_zoom.axis("off")
        self.fig.canvas.draw()

    def _adjust_point_in_zoom(self, event):
        if self.zoom_image_coords:
            x0, y0, _, _ = self.zoom_image_coords
            new_x, new_y = x0 + event.xdata, y0 + event.ydata
            self.points.append((new_x, new_y))
            self.ax_zoom.plot(event.xdata, event.ydata, "bo", markersize=10)
            self.ax.plot(new_x, new_y, "bo", markersize=5)  # Mark the adjusted point
            self.point_count += 1
            if self.point_count < self.num_points:
                self.ax.set_title(
                    f"Image: Select Point {self.point_count} of {self.num_points}"
                )
            else:
                self.ax.set_title(
                    'Selection Completed - Press "r" to reset or "q" to quit'
                )
            self.zoom_image_coords = None
            self.fig.canvas.draw()

    def _onkey(self, event):
        if event.key == "r":
            self._reset()
        elif event.key == "q":
            plt.close(self.fig)

    def _reset(self):
        """Reset the point selection."""
        self.points = []
        self.point_count = 0
        self.ax.clear()
        self.ax_zoom.clear()
        self.ax.imshow(self.image)
        self.ax.set_title(f"Image: Select Point 0 of {self.num_points}")
        self.ax.axis("off")
        self._init_zoom()
        self.fig.canvas.draw()

    def reset(self):
        self._reset()

    def get_points(self):
        return self.points
