"""A GUI for setting parameters"""

from __future__ import absolute_import

from Tkinter import *
from ttk import *

__all__ = ['show_config_gui']


def show_config_gui(parameters):
    """Show the GUI for configuration

    :arg parameters: Parameters as a :class:`firedrake.parameters.Parameters` class
    """
    from firedrake import Parameters

    if not isinstance(parameters, Parameters):
        raise TypeError("Expected Type: Parameters")

    def load_json():
        """Creates a dialog box prompting json input file, then import
        parameters from json to current parameters
        """
        import tkFileDialog
        filename = tkFileDialog.askopenfilename()
        import_params_from_json(parameters, filename)
        refresh_params(parameters)

    def save_json():
        """Creates a dialog box prompting json output file, then validates
        current parameters and export parameters to json file if validated
        """
        import tkFileDialog
        if save_params():
            filename = tkFileDialog.asksaveasfilename()
            export_params_to_json(parameters, filename)

    def save_and_quit():
        """Save the current input into current parameters and close the window
        """
        if save_params():
            root.destroy()

    def save_params():
        """Save the current parameters from input, pop out a message box if
        there is an error
        """
        try:
            parsed_dict = parse_input_dict(parameters)
            load_from_dict(parameters, parsed_dict)
            return True
        except ValueError as e:
            from tkMessageBox import showinfo
            showinfo(title="Error", message=e.message, icon="error", parent=root)
            return False

    def refresh_params(parameters):
        """Refresh the GUI values from a given source

        :arg paramaters: parameters as a
            :class:`firedrake.parameters.Parameters` class
        """
        for key in parameters.keys():
            if isinstance(parameters[key], Parameters):
                refresh_params(parameters[key])
            else:
                key.variable.set(str(parameters[key]))

    def parse_input_dict(parameters):
        """Generate a dictionary of values from variables

        :arg parameters: Parameters as :class:`firedrake.parameters`
        :raises ValueError: when input value is invalid
        """
        from firedrake import Parameters

        parsed_dict = {}
        for key in parameters.keys():
            if isinstance(parameters[key], Parameters):
                parsed_dict[key] = parse_input_dict(parameters[key])
            else:
                if hasattr(key, "variable"):
                    str_val = key.variable.get()
                    if not key.validate(str_val):
                        raise ValueError("Invalid value for parameter %s" % key)
                    parsed_dict[key] = key.type.parse(str_val)
        return parsed_dict

    def create_ui_element(parameters, parent, key, row):

        def create_true_false_button(parent, variable, row):
            button_true = Radiobutton(parent, text='True',
                                      variable=variable, value="True")
            button_true.grid(column=2, row=row, sticky=(E))
            button_false = Radiobutton(parent, text='False',
                                       variable=variable, value="False")
            button_false.grid(column=3, row=row, sticky=(E))

        def create_options_drop_list(parent, variable, default, options, row):
            drop_list = OptionMenu(parent, variable, default, *options)
            drop_list.grid(column=2, columnspan=2, row=row, sticky=(E))

        def create_text_entry(parent, variable, row):
            label_val = Entry(parent, textvariable=variable)
            label_val.grid(column=2, columnspan=2, row=row, sticky=(E))

        def create_config_box_or(parent, key, row):
            def config_or_type():
                def callback():
                    window = Toplevel(root)
                    type_idx = IntVar()
                    var = [StringVar() for t in key.type.types]
                    sub_row = 0

                    # infer current type from value and clear current type
                    key.type.parse(key.variable.get())
                    curr_type_idx = key.type.types.index(key.type.curr_type)
                    type_idx.set(curr_type_idx)
                    var[curr_type_idx].set(str(key.variable.get()))

                    for type in key.type.types:
                        type_selector = Radiobutton(window, text=str(type),
                                                    variable=type_idx,
                                                    value=sub_row)
                        type_selector.grid(column=1, row=sub_row)
                        if isinstance(type, BoolType):
                            create_true_false_button(window, var[sub_row],
                                                     sub_row)
                        elif isinstance(type, StrType) and type.options != []:
                            create_options_drop_list(window, var[sub_row],
                                                     type.options[0],
                                                     type.options, sub_row)
                        else:
                            create_text_entry(window, var[sub_row], sub_row)
                        sub_row += 1
                    key.type.clear_curr_type()

                    def save():
                        def callback():
                            key.variable.set(var[type_idx.get()].get())
                            key.type.curr_type = type_idx.get()
                            window.destroy()
                        return callback

                    ok = Button(window, text='OK', command=save())
                    ok.grid(column=2, row=sub_row)
                return callback

            button = Button(parent, text='Configure',
                            command=config_or_type())
            button.grid(column=2, columnspan=2, row=row, sticky=(E))

        from firedrake.parameters import BoolType, OrType, StrType
        if isinstance(key.type, BoolType):
            create_true_false_button(parent, key.variable, row_count)
        elif isinstance(key.type, StrType) and key.type.options != []:
            create_options_drop_list(parent, key.variable, parameters[key],
                                     key.type.options, row_count)
        elif isinstance(key.type, OrType):
            create_config_box_or(parent, key, row_count)
        else:
            create_text_entry(parent, key.variable, row_count)

    def generate_input(parameters, labelframe):
        """Generates GUI elements for parameters inside a label frame

        :arg parameters: Parameters as :class:`firedrake.parameters`
        :arg labelframe: :class:`ttk.Labelframe` to place the GUI elements
        """
        global row_count
        keys = sorted(parameters.keys())
        for key in keys:
            row_count += 1
            if isinstance(parameters[key], Parameters):
                subframe = Labelframe(labelframe, text=key,
                                      padding='3 3 12 12')
                subframe.grid(column=1, columnspan=4,
                              row=row_count, sticky=(W, E))
                subframe.columnconfigure(1, weight=1)
                subframe.rowconfigure(0, weight=1)
                key.variable = {}
                generate_input(parameters[key], subframe)
            else:
                label_key = Label(labelframe, text=key)
                label_key.grid(column=1, row=row_count, sticky=(W))
                key.variable = StringVar()
                key.variable.set(str(parameters[key]))
                create_ui_element(parameters, labelframe, key, row_count)

                def help_box(key):
                    def click():
                        from tkMessageBox import showinfo
                        showinfo(title="Help", message=key.help, parent=root)
                    return click

                help_button = Button(labelframe, text='Help',
                                     command=help_box(key))
                help_button.grid(column=4, row=row_count, sticky=(E))

    def configure_frame(event):
        """Callback for frame resizing"""
        size = (mainframe.winfo_reqwidth(), mainframe.winfo_reqheight())
        canvas.config(scrollregion="0 0 %s %s" % size)
        if mainframe.winfo_reqwidth() != canvas.winfo_width():
            canvas.config(width=mainframe.winfo_reqwidth())
        canvas.config(height=mainframe.winfo_reqheight())

    def configure_canvas(event):
        """Callback for canvas resizing"""
        if mainframe.winfo_reqwidth() != canvas.winfo_width():
            canvas.itemconfigure(frame_id, width=canvas.winfo_width())

    global root
    root = Tk()
    root.title("Configure")

    canvas = Canvas(root, borderwidth=0)

    mainframe = Frame(canvas, padding='3 3 12 12')
    mainframe.grid(row=0, column=0, sticky=(N, W, E, S))
    mainframe.columnconfigure(0, weight=1)
    mainframe.rowconfigure(0, weight=1)

    scrollbar = Scrollbar(root, orient=VERTICAL, command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)

    scrollbar.pack(side=RIGHT, fill=Y)
    canvas.pack(side=LEFT, fill=BOTH, expand=True)
    frame_id = canvas.create_window(0, 0, window=mainframe, anchor=NW)

    global row_count
    row_count = 1

    generate_input(parameters, mainframe)

    row_count += 1
    button_load = Button(mainframe,
                         text="Load from File",
                         command=load_json)
    button_load.grid(column=1, row=row_count, sticky=W)
    button_save = Button(mainframe, text="Save to File", command=save_json)
    button_save.grid(column=3, row=row_count, sticky=S)
    button_quit = Button(mainframe, text="Save and Quit",
                         command=save_and_quit)
    button_quit.grid(column=4, row=row_count, sticky=E)

    mainframe.bind('<Configure>', configure_frame)
    canvas.bind('<Configure>', configure_canvas)

    root.mainloop()


def export_params_to_json(parameters, filename):
    """Export parameters to a JSON file

    :arg parameters: Parameters as a :class:`firedrake.parameters.Parameters`
        class
    :arg filename: File name of the output file
    """
    import json

    if filename == '':
        return
    output_file = open(filename, 'w')
    output_file.write(json.dumps(parameters))
    output_file.close()


def import_params_from_json(parameters, filename):
    """Import parameters from a JSON file

    :arg parameters: Parameters as a :class:`firedrake.parameters.Parameters`
        class
    :arg filename: File name of the input file
    """
    import json

    if filename == '':
        return
    input_file = open(filename, 'r')
    dictionary = json.loads(input_file.read())
    input_file.close()
    load_from_dict(parameters, dictionary)
    return parameters


def load_from_dict(parameters, dictionary):
    """Merge the parameters in a dictionary into Parameters class

    :arg parameters: Parameters to be merged into as a
        :class:`firedrake.parameters.Parameters` class
    :arg dictionary: Dictionary of parameters to be merged
    """
    from firedrake import Parameters
    from firedrake.logging import warning

    for k in dictionary:
        if k in parameters:
            if isinstance(parameters[k], Parameters):
                load_from_dict(parameters[k], dictionary[k])
            else:
                try:
                    if isinstance(dictionary[k], unicode):
                        # change unicode type to str type
                        parameters[k] = dictionary[k].encode('ascii',
                                                             'ignore')
                    else:
                        parameters[k] = dictionary[k]
                except ValueError as e:
                    from tkMessageBox import showinfo
                    showinfo(title="Error", message=e.message, icon="error",
                             parent=root)
        else:
            warning(k + ' is not in the parameters and ignored')
