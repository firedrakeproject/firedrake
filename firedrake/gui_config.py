"""A GUI for setting parameters"""

from __future__ import absolute_import

from Tkinter import *
from ttk import *

__all__ = ['show_config_gui']


def show_config_gui(parameters):
    from firedrake import Parameters

    if not isinstance(parameters, Parameters):
        raise TypeError("Expected Type: Parameters")

    def load_json():
        import tkFileDialog
        filename = tkFileDialog.askopenfilename()
        import_params_from_json(parameters, filename)

    def save_json():
        import tkFileDialog
        save_params()
        filename = tkFileDialog.asksaveasfilename()
        export_params_to_json(parameters, filename)

    def save_and_quit():
        save_params()
        root.destroy()

    def save_params():
        pass

    def generate_input(parameters, labelframe):
        global row_count
        keys = sorted(parameters.keys())
        for key in keys:
            row_count += 1
            if isinstance(parameters[key], Parameters):
                subframe = Labelframe(labelframe, text=key, padding='3 3 12 12')
                subframe.grid(column=1, columnspan=3, row=row_count, sticky=(W, E))
                subframe.columnconfigure(1, weight=1)
                subframe.rowconfigure(0, weight=1)
                generate_input(parameters[key], subframe)
            else:
                label_key = Label(labelframe, text=key)
                label_key.grid(column=1, row=row_count, sticky=(W))
                label_val = Label(labelframe, text=parameters[key])
                label_val.grid(column=3, row=row_count, sticky=(E))

    root = Tk()
    root.title("Configure")

    mainframe = Frame(root, padding='3 3 12 12')
    mainframe.grid(row=0, column=0, sticky=(N, W, E, S))
    mainframe.columnconfigure(0, weight=1)
    mainframe.rowconfigure(0, weight=1)

    global row_count
    row_count = 1

    generate_input(parameters, mainframe)

    row_count += 1
    button_load = Button(mainframe,
                         text="Load from File",
                         command=load_json)
    button_load.grid(column=1, row=row_count, sticky=W)
    button_save = Button(mainframe, text="Save to File", command=save_json)
    button_save.grid(column=2, row=row_count, sticky=S)
    button_quit = Button(mainframe, text="Save and Quit", command=save_and_quit)
    button_quit.grid(column=3, row=row_count, sticky=E)

    root.mainloop()


def export_params_to_json(parameters, filename):
    import json

    if filename == '':
        return
    output_file = open(filename, 'w')
    output_file.write(json.dumps(parameters))
    output_file.close()


def import_params_from_json(parameters, filename):
    import json

    if filename == '':
        return
    input_file = open(filename, 'r')
    dictionary = json.loads(input_file.read())
    input_file.close()
    load_from_dict(parameters, dictionary)
    return parameters


def load_from_dict(parameters, dictionary):
    from firedrake import Parameters
    from firedrake.logging import warning

    for k in dictionary:
        if k in parameters:
            if isinstance(parameters[k], Parameters):
                load_from_dict(parameters[k], dictionary[k])
            else:
                if isinstance(dictionary[k], unicode):
                    # change unicode type to str type
                    parameters[k] = dictionary[k].encode('ascii', 'ignore')
                else:
                    parameters[k] = dictionary[k]
        else:
            warning(k + ' is not in the parameters and ignored')
