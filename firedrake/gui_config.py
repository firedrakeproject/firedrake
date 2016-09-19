"""A GUI for setting parameters"""

from __future__ import absolute_import

from Tkinter import *
from ttk import *

__all__ = ['show_config_gui',
           "export_params_to_json",
           "import_params_from_json"]


def show_config_gui(parameters, visible_level=0):
    """Show the GUI for configuration

    :arg parameters: Parameters as a :class:`firedrake.parameters.Parameters` class
    :arg visible_level: visible level of parameters displayed. Parameter with
        higher than given value will not be displayed.
    """
    from firedrake import Parameters

    if not isinstance(parameters, Parameters):
        raise TypeError("Expected Type: Parameters")

    def load_json():
        """Callback for load JSON button

        Create a dialog box prompting json input file, then import
        parameters from json to current parameters.
        """
        import tkFileDialog
        filename = tkFileDialog.askopenfilename()
        try:
            import_params_from_json(parameters, filename)
            refresh_params(parameters)
        except ValueError as e:
            from tkMessageBox import showinfo
            showinfo(title="Error", message=e.message, icon="error",
                     parent=root)

    def save_json():
        """Callback for save JSON button

        Create a dialog box prompting json output file, then validate
        current parameters and export parameters to json file if
        validated.
        """
        import tkFileDialog
        if save_params():
            filename = tkFileDialog.asksaveasfilename()
            export_params_to_json(parameters, filename)

    def save_and_quit():
        """Callback for save and quit button

        Save the current input into current parameters and close the
        window.
        """
        if save_params():
            root.destroy()

    def save_params():
        """Save current inputs to parameters

        Save the current parameters from input, pop out a message box if
        there is an error.
        """
        try:
            parsed_dict = parse_input_dict(parameters)
            load_from_dict(parameters, parsed_dict)
            return True
        except ValueError as e:
            from tkMessageBox import showinfo
            showinfo(title="Error", message=e.message,
                     icon="error", parent=root)
            return False

    def refresh_params(parameters):
        """Refresh the GUI values from a given source

        :arg paramaters: parameters as a
            :class:`firedrake.parameters.Parameters` class
        """
        parameters = parameters.unwrapped_dict(visible_level)
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
                        raise ValueError("Invalid value for parameter %s" %
                                         key)
                    parsed_dict[key] = key.type.parse(str_val)
        return parsed_dict

    def create_ui_element(parent, key, row):
        """Create the UI element for a key.

        :arg parent: the parent of UI element
        :arg key: the key of the input as
            :class:`firedrake.parameters.TypedKey`
        :arg row: the row number of the UI element
        """
        def create_true_false_button(parent, variable, row):
            """Create two Radiobuttons for boolean input type."""
            button_true = Radiobutton(parent, text='True',
                                      variable=variable, value="True")
            button_true.grid(column=2, row=row, sticky=(E))
            button_false = Radiobutton(parent, text='False',
                                       variable=variable, value="False")
            button_false.grid(column=3, row=row, sticky=(E))
            return [button_true, button_false]

        def create_options_drop_list(parent, variable, default, options, row):
            """Create an optionmenu for enum input type.

            (represented as string).
            """
            drop_list = OptionMenu(parent, variable, default, *options)
            drop_list.grid(column=2, columnspan=2, row=row, sticky=(E))
            return [drop_list]

        def create_text_entry(parent, variable, row):
            """Create a text entry for input."""
            label_val = Entry(parent, textvariable=variable)
            label_val.grid(column=2, columnspan=2, row=row, sticky=(E))
            return [label_val]

        def create_config_box_or(parent, variable, key_type, row):
            """Create a configure button for OrType.

            On clicking, pop out a config dialog for choosing multiple input
            types.
            """
            def config_or_type():
                def callback():
                    window = Toplevel(root)
                    type_idx = IntVar()
                    var = [StringVar() for t in key_type.types]
                    sub_row = 0

                    # infer current type from value and clear current type
                    key_type.parse(variable.get())
                    curr_type_idx = key_type.types.index(key_type.curr_type)
                    type_idx.set(curr_type_idx)
                    var[curr_type_idx].set(str(variable.get()))

                    for type in key_type.types:
                        type_selector = Radiobutton(window, text=str(type),
                                                    variable=type_idx,
                                                    value=sub_row)
                        type_selector.grid(column=1, row=sub_row)
                        generate_ui_type_selector(window, type,
                                                  var[sub_row], sub_row)
                        sub_row += 1
                    key_type.clear_curr_type()

                    def save():
                        def callback():
                            variable.set(var[type_idx.get()].get())
                            key_type.curr_type = type_idx.get()
                            window.destroy()

                        return callback

                    ok = Button(window, text='OK', command=save())
                    ok.grid(column=2, row=sub_row)

                return callback

            button = Button(parent, text='Configure',
                            command=config_or_type())
            button.grid(column=2, columnspan=2, row=row, sticky=(E))
            return button

        def create_config_box_list(parent, variable, key_type, row):
            """Create a configure button for ListType.

            On clicking, pop out a config dialog box for list inputs.
            """
            def config_list_type():
                def callback():
                    window = Toplevel(root)
                    list_box = Listbox(window, selectmode=SINGLE, height=10)
                    list_box.grid(row=1, columnspan=3,
                                  column=1, sticky=(N, S, W, E))

                    for elem in parameters[key]:
                        list_box.insert(END, str(elem))

                    def save():
                        def callback():
                            values = list_box.get(0, END)
                            lst = []
                            for value in values:
                                if key_type.elem_type.validate(value):
                                    lst.append(key_type.elem_type.parse(value))
                            if key_type.validate(lst):
                                variable.set(str(lst))
                                window.destroy()
                            else:
                                from tkMessageBox import showinfo
                                showinfo(title="Error",
                                         message="Input is invalid")

                        return callback

                    def add_elem():
                        def callback():
                            str_val = new_var.get()
                            if key_type.elem_type.validate(str_val):
                                list_box.insert(END,
                                                str(key_type.elem_type
                                                    .parse(str_val)))
                            else:
                                from tkMessageBox import showinfo
                                showinfo(title="Error",
                                         message="Invalid new value",
                                         parent=window)

                        return callback

                    def del_elem():
                        def callback():
                            list_box.delete(ANCHOR)

                        return callback

                    new_var = StringVar()
                    generate_ui_type_selector(window, key_type.elem_type,
                                              new_var, 2)
                    label = Label(window, text="New Value:")
                    label.grid(column=1, row=2)
                    ok = Button(window, text='OK', command=save())
                    ok.grid(column=2, row=3)
                    add = Button(window, text='+', command=add_elem())
                    add.grid(column=1, row=3)
                    minus = Button(window, text='-', command=del_elem())
                    minus.grid(column=3, row=3)
                    help_text = Label(window,
                                      text="Enter a new value or \
click on config button, then click + button to add into list.\n Click on an \
item, then click - button to delete from list",
                                      wraplength=250)
                    help_text.grid(column=1, columnspan=3, row=4)

                return callback

            button = Button(parent, text='Configure',
                            command=config_list_type())
            button.grid(column=2, columnspan=2, row=row, sticky=(E))
            return [button]

        def generate_ui_type_selector(parent, type, variable, row):
            """Create UI element for input.

            Select the UI element creation function according to the type
            of the key given.
            """
            from firedrake.parameters import BoolType, OrType, StrType, ListType
            if isinstance(type, BoolType):
                return create_true_false_button(parent, variable, row)
            elif isinstance(type, StrType) and type.options != []:
                return create_options_drop_list(parent, variable,
                                                variable.get(),
                                                type.options, row)
            elif isinstance(type, OrType):
                return create_config_box_or(parent, variable, type, row)
            elif isinstance(type, ListType):
                return create_config_box_list(parent, variable, type, row)
            else:
                return create_text_entry(parent, variable, row)

        return generate_ui_type_selector(parent, key.type,
                                         key.variable, row_count)

    def generate_input(parameters, labelframe):
        """Generate GUI elements for parameters inside a label frame

        :arg parameters: Parameters as :class:`firedrake.parameters`
        :arg labelframe: :class:`ttk.Labelframe` to place the GUI elements
        """
        global row_count
        parameters = parameters.unwrapped_dict(visible_level)
        keys = sorted(parameters.keys())
        ui_elems = []
        for key in keys:
            row_count += 1
            if isinstance(parameters[key], Parameters):
                def show_hide_labelframe(elems, frame):

                    def callback():
                        if not frame.is_hidden:
                            map(lambda x: x.grid_remove(), elems)
                            frame.config(height=30)
                            frame.is_hidden = True
                            configure_canvas(None)
                            configure_frame(None)
                        else:
                            map(lambda x: x.grid(), elems)
                            frame.is_hidden = False
                            configure_canvas(None)
                            configure_frame(None)
                    return callback

                subframe = Labelframe(labelframe, text=key,
                                      padding='3 3 12 12')
                subframe.grid(column=1, columnspan=4,
                              row=row_count, sticky=(W, E))
                subframe.is_hidden = False
                ui_elems.append(subframe)
                subframe.columnconfigure(1, weight=1)
                subframe.rowconfigure(0, weight=1)
                key.variable = {}
                sub_elems = generate_input(parameters[key], subframe)
                ui_elems.extend(sub_elems)
                show_hide = Button(labelframe, text='Show/Hide',
                                   command=show_hide_labelframe(sub_elems,
                                                                subframe))
                show_hide.grid(column=4, row=row_count, sticky=NE)
                ui_elems.append(show_hide)
            else:
                label_key = Label(labelframe, text=key)
                label_key.grid(column=1, row=row_count, sticky=(W))
                ui_elems.append(label_key)
                key.variable = StringVar()
                key.variable.set(str(parameters[key]))
                ui_elems.extend(create_ui_element(labelframe, key, row_count))

                def help_box(key):
                    def click():
                        from tkMessageBox import showinfo
                        showinfo(title="Help", message=key.help, parent=root)

                    return click

                help_button = Button(labelframe, text='Help',
                                     command=help_box(key))
                help_button.grid(column=4, row=row_count, sticky=(E))
                ui_elems.append(help_button)
        return ui_elems

    def configure_frame(event):
        """Callback for frame resizing"""
        size = (mainframe.winfo_reqwidth(), mainframe.winfo_reqheight())
        canvas.config(scrollregion="0 0 %s %s" % size)
        if mainframe.winfo_reqwidth() != canvas.winfo_width():
            canvas.config(width=mainframe.winfo_reqwidth())
        canvas.config(height=mainframe.winfo_reqheight())
        root.geometry("%sx%s" % size)

    def configure_canvas(event):
        """Callback for canvas resizing"""
        if mainframe.winfo_reqwidth() != canvas.winfo_width():
            canvas.itemconfigure(frame_id, width=canvas.winfo_width())

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
    json.dump(parameters.unwrapped_dict(-1), output_file)
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
    dictionary = json.load(input_file)
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
                val = dictionary[k]
                if isinstance(val, unicode):
                    # change unicode type to str type
                    val = val.encode('ascii', 'ignore')
                    val = parameters.get_key(k).type.parse(val)
                parameters[k] = parameters.get_key(k).wrap(val)
        else:
            warning(k + ' is not in the parameters and ignored')
