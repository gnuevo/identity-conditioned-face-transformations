import torch

class ValidationImageGrid(object):
    """Creates a grid of images according to the validation fashion:
    + the first row contains the conditioning images
    + the first column contains the input images
    + the remaining positions x(a,b) contain input(a) | conditioning(b)
    """

    def __init__(self, num_cond):
        self.current_row = [] # buffer that contains the current row
        self.input_column = [] # column with the original input images
        self.cond_row = [] # row with the original conditioning images
        self.all_rows = [] # contains all rows
        self.index = 0
        self.num_cond = num_cond

    def add_images(self, A, B, C):
        """Adds images to the grid

        Args:
            A: batch of input images
            B: batch of conditioning images
            C: batch of output (transformed) images
        """
        for a, b, c in zip(A, B, C):
            self.current_row.append(c.to('cpu'))

            if self.index % self.num_cond == 0:
                self.input_column.append(a.to('cpu'))

            if self.index < self.num_cond:
                self.cond_row.append(b.to('cpu'))

            self.index += 1
            if self.index % self.num_cond == 0:
                self.all_rows.append(self.current_row)
                self.current_row = []

    def compose(self):
        """Generates a tensor containing all the validation images

        Returns: tensor containing all the validation images
        """
        black = torch.zeros_like(self.cond_row[0])
        # concatenate individual images into groups
        first_row = torch.cat([black, *self.cond_row], dim=2)
        body_rows = [torch.cat(row, dim=2) for row in self.all_rows]
        body = torch.cat(body_rows, dim=1)
        input_column = torch.cat(self.input_column, dim=1)
        input_and_body = torch.cat([input_column, body], dim=2)
        total = torch.cat([first_row, input_and_body], dim=1)
        return total