from src.model.MLP import MLP
from src.ui.draw_digit_ui import DrawDigitUI

if __name__ == '__main__':
    mlp = MLP()
    mlp.load_model()

    ui = DrawDigitUI()
    ui.build_ui(mlp)
    ui.run()


