import numpy as np
import pandas as pd
import pickle

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.uix.widget import Widget
from kivy.uix.screenmanager import ScreenManager, Screen

class MainScreen(Screen):
    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
        
        self.layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        self.label = Label(text="Please give Input Request", font_size='20sp', size_hint=(1, 0.2))
        self.layout.add_widget(self.label)

        self.btn_browse = Button(text="Browse Request Files", size_hint=(1, 0.2), on_press=self.browse_files)
        self.layout.add_widget(self.btn_browse)

        self.btn_start = Button(text="Start Analyzing Request", size_hint=(1, 0.2), on_press=self.start_analysis)
        self.layout.add_widget(self.btn_start)

        self.btn_exit = Button(text="Exit", size_hint=(1, 0.2), on_press=App.get_running_app().stop)
        self.layout.add_widget(self.btn_exit)

        self.add_widget(self.layout)

    def browse_files(self, instance):
        content = FileChooserIconView()
        content.bind(on_submit=self.file_selected)

        self.popup = Popup(title="Select a CSV File",
                           content=content,
                           size_hint=(0.9, 0.9))
        self.popup.open()

    def file_selected(self, filechooser, selection, touch):
        if selection:
            self.filepath = selection[0]
            self.label.text = f"File Opened: {self.filepath}"
            self.popup.dismiss()

    def start_analysis(self, instance):
        if hasattr(self, 'filepath'):
            print("Process Started")
            dataset = pd.read_csv(self.filepath)
            dataset = dataset.dropna(how="any")
            print(dataset.info())

            X = dataset.iloc[:, 6:16].values

            # Load the model from disk
            model = pickle.load(open('knnpickle_file', 'rb'))
            ypred = model.predict(X)
            ypred = ypred.round()
            print(ypred)

            if ypred[0] == 0:
                self.label.text = "There is Everything Normal, Nothing to Worry !!"
                self.label.color = [0, 1, 0, 1]  # Green color
            else:
                self.label.text = "Possibility of DDOS Attack, Better take the Precautions!!"
                self.label.color = [1, 0, 0, 1]  # Red color
        else:
            self.label.text = "Please select a file first."
            self.label.color = [1, 1, 0, 1]  # Yellow color for warning

class DdosApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(MainScreen(name='main'))
        return sm

if __name__ == '__main__':
    DdosApp().run()
