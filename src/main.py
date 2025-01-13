from model.CNN import CNN
from ui.draw_digit_ui import DrawDigitUI

if __name__ == '__main__':
    model = CNN()

    model.load_model()
    print(model.get_model().summary())

    ui = DrawDigitUI()
    ui.build_ui(model)
    ui.run()

