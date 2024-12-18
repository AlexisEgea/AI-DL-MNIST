from src.model.MLP import MLP
from src.model.CNN import CNN
from src.ui.draw_digit_ui import DrawDigitUI

if __name__ == '__main__':
    # model = MLP()
    model = CNN()
    model.load_model()
    print(model.model.summary())

    ui = DrawDigitUI()
    ui.build_ui(model)
    ui.run()


