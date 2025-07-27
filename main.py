from FontEnd import gui_app
import warnings
import logging

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    gui_app.main()