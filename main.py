import dearpygui.dearpygui as dpg
import dearpygui.demo as demo
import asyncio

import numpy as np

from thermal_functions import load_thermal_file, load_thermal_and_mic_file, get_size_from_mic_file, extract_gradient_and_temp_timestamped_NUTS, extract_non_max_steady_state_from_diff, calibrate_gradient, calculate_attenuation_mesh, calculate_absorbtion_mesh, press_from_gradient_mesh, press_from_grad_fit, calibrate_steady_state_naive
import copy

import threading

import flircamera

# TODO: allow selection of noise-reduction parameters

class DataHolder:
    timestamps = []
    start_indices = []
    mic_values = []
    mic_peaks = []
    ambients = []
    thermals_s = []
    thermals_g = []
    thermals_s_nr = []
    thermals_g_nr = []

    group_name = []
    name = []

    gradients = []
    gradients_maxs = []
    gradients_temp = []
    saturations = []
    saturations_maxs = []

    fit_grad_coeffs = []
    fit_saturation_coeffs = {}


class MeasurementDataHolder:
    timestamps = []
    start_indices = []
    ambients = []
    thermals = []
    thermals_nr = []

    name = []

    gradients = []
    gradients_maxs = []
    gradients_temp = []
    saturations = []
    saturations_maxs = []

    pressures = []

    converted_indices = []


# Globals, should probs try get rid of them!
#loop = asyncio.get_event_loop()

data_holder = DataHolder()
data_holder_measurement = MeasurementDataHolder()

tabs = ["tab 1"]

validation_selected_index = 0

camera = None
recording = False

dpg.create_context()

with dpg.font_registry():
    # first argument ids the path to the .ttf or .otf file
    default_font = dpg.add_font("NotoSerifCJKjp-Medium.otf", 20)
    second_font = dpg.add_font("NotoSerifCJKjp-Medium.otf", 10)

dpg.bind_font(default_font)

dpg.create_viewport(title="Sound Field Measurement via Thermography", width=1650, height=1000,small_icon="icon.ico",large_icon="icon.ico",decorated=True)

dpg.setup_dearpygui()

dpg.setup_registries()


def add_tab_thermal_data(sender, app_data, user_data):

    tabs.append(f"tab {len(tabs)+1}")

    with dpg.tab(parent="thermal_data_tab_bar", label=tabs[-1],before="thermal_data_add_tab_button") as tab:
        dpg.add_input_text(hint="Rename Data Group here",callback=rename_data_group, user_data=tab)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Choose Thermal Files", user_data=fd, callback=lambda s, a, u: dpg.configure_item(u, show=True,user_data=(str(tab)+"_list_box_therm",str(tab))))
            dpg.add_button(label="Choose Microphone Files", user_data=fd2, callback=lambda s, a, u: dpg.configure_item(u, show=True,user_data=(str(tab)+"_list_box_mic",str(tab))))
        with dpg.group(horizontal=True,width=600):
            dpg.add_listbox(tag=str(tab)+"_list_box_therm")
            dpg.add_listbox(tag=str(tab)+"_list_box_mic")
        dpg.add_button(label="Remove Data Group", callback=remove_data_group, user_data=tab)

        dpg.set_value("thermal_data_tab_bar", tab)


    with dpg.tab(parent="thermal_data_tab_bar2", label=tabs[-1]) as tab2:
        with dpg.group(horizontal=True,width=600):
            dpg.add_listbox(tag=str(tab2)+"_list_box_therm2")
            dpg.add_listbox(tag=str(tab2)+"_list_box_mic2")
        with dpg.group(horizontal=True):
            dpg.add_button(label="View Selected Data",enabled=False, tag=str(tab2)+"view_converted_data_button_validation",callback=view_thermal_measurement_validataion)


def view_converted_measurement(sender, app_data, user_data):

    selected_file_index = dpg.get_item_configuration("list_box_measurement_converted")["items"].index(dpg.get_value("list_box_measurement_converted"))

    data = data_holder_measurement.pressures[selected_file_index]

    if selected_file_index not in data_holder_measurement.converted_indices:
        return

    if dpg.does_item_exist("converted_data_plot"):
        dpg.set_value("converted_data_plot", [data]) 
        dpg.configure_item("converted_data_plot",cols=data.shape[1],rows=data.shape[0],scale_min=0,scale_max=np.max(data))
        dpg.configure_item("converted_data_legend",min_scale=0,max_scale=np.max(data))
    else:
        dpg.add_heat_series(data,cols=data.shape[1],rows=data.shape[0], parent="connverted_y_axis",scale_min=0,scale_max=np.max(data),format="",tag="converted_data_plot")
        dpg.bind_colormap("converted_data_heat", dpg.mvPlotColormap_Plasma)
        dpg.configure_item("converted_data_legend",min_scale=0,max_scale=np.max(data))

    dpg.set_item_width("converted_data_heat",int(dpg.get_item_height("converted_data_heat")* (data.shape[1]/data.shape[0])))


def view_thermal_measurement_validataion(sender, app_data, user_data):

    global validation_selected_index

    active_tab = dpg.get_value("thermal_data_tab_bar2")

    selected_file = dpg.get_value(str(active_tab)+"_list_box_therm2")

    for index, (therm_file, _) in enumerate(data_holder.name):
        if therm_file == selected_file:
            break

    validation_selected_index = index

    data = data_holder.thermals_g_nr[index][0]

    if dpg.does_item_exist("validation_data_plot"):
        dpg.set_value("validation_data_plot", [data]) 
        dpg.configure_item("validation_data_plot",cols=data.shape[1],rows=data.shape[0],scale_min=np.min(data),scale_max=np.max(data))
        dpg.configure_item("validation_data_legend",min_scale=np.min(data),max_scale=np.max(data))
    else:
        dpg.add_heat_series(data,cols=data.shape[1],rows=data.shape[0], parent="validation_y_axis",scale_min=np.min(data),scale_max=np.max(data),format="",tag="validation_data_plot")
        dpg.bind_colormap("validation_data_heat", dpg.mvPlotColormap_Plasma)
        dpg.configure_item("validation_data_legend",min_scale=np.min(data),max_scale=np.max(data))

    dpg.set_item_width("validation_data_heat",int( dpg.get_item_height("validation_data_heat") * (data.shape[1]/data.shape[0])))
    dpg.set_item_width("validation_time_slider",int( dpg.get_item_height("validation_data_heat") * (data.shape[1]/data.shape[0])) )

    dpg.configure_item("validation_time_slider",max_value=data_holder.thermals_g_nr[index].shape[0]-1)

    dpg.show_item("validation_time_slider")



    gradients = data_holder.gradients[index]
    gradients[gradients <= 0] = 0.0000000000000001

    # convert gradients to pressure

    if dpg.get_value("gradient_model_selector") ==  "Physical Model":
        data = press_from_mesh(gradients)
    else:
        data = press_from_grad_fit(gradients, data_holder.fit_grad_coeffs[1], data_holder.fit_grad_coeffs[2])

    gradient_pressure = data

    if dpg.does_item_exist("validation_grad_plot"):
        dpg.set_value("validation_grad_plot", [data]) 
        dpg.configure_item("validation_grad_plot",cols=data.shape[1],rows=data.shape[0],scale_min=np.min(data),scale_max=np.max(data))
        dpg.configure_item("validation_grad_legend",min_scale=np.min(data),max_scale=np.max(data))
    else:
        dpg.add_heat_series(data,cols=data.shape[1],rows=data.shape[0], parent="validation_grad_y_axis",scale_min=np.min(data),scale_max=np.max(data),format="",tag="validation_grad_plot")
        dpg.bind_colormap("validation_grad_heat", dpg.mvPlotColormap_Plasma)
        dpg.configure_item("validation_grad_legend",min_scale=np.min(data),max_scale=np.max(data))

    dpg.set_item_width("validation_grad_heat",int( dpg.get_item_height("validation_grad_heat") * (data.shape[1]/data.shape[0])))


    saturations = data_holder.saturations[index]
    saturations[saturations <= 0] = 0.0000000000000001

    # convert saturations to pressure

    if dpg.get_value("saturation_model_selector") ==  "Physical Model":
        data = data_holder.fit_saturation_coeffs["quadratic_emission_streaming"](saturations) #TODO: IMPLEMENT THIS
    else:
        data = data_holder.fit_saturation_coeffs["quadratic_emission_streaming"](saturations) 

    saturation_pressure = data

    if dpg.does_item_exist("validation_sat_plot"):
        dpg.set_value("validation_sat_plot", [data]) 
        dpg.configure_item("validation_sat_plot",cols=data.shape[1],rows=data.shape[0],scale_min=np.min(data),scale_max=np.max(data))
        dpg.configure_item("validation_sat_legend",min_scale=np.min(data),max_scale=np.max(data))
    else:
        dpg.add_heat_series(data,cols=data.shape[1],rows=data.shape[0], parent="validation_sat_y_axis",scale_min=np.min(data),scale_max=np.max(data),format="",tag="validation_sat_plot")
        dpg.bind_colormap("validation_sat_heat", dpg.mvPlotColormap_Plasma)
        dpg.configure_item("validation_sat_legend",min_scale=np.min(data),max_scale=np.max(data))

    dpg.set_item_width("validation_sat_heat",int( dpg.get_item_height("validation_sat_heat") * (data.shape[1]/data.shape[0])))

    data = copy.deepcopy(data_holder.mic_values[index]) # if we don't deep copy for some reason we get a crash with no trace on plotting

    data[data <= 0] = 0.0000000000000001


    if dpg.does_item_exist("validation_mic_plot"):
        dpg.set_value("validation_mic_plot", [data]) 
        dpg.configure_item("validation_mic_plot",cols=data.shape[1],rows=data.shape[0],scale_min=np.min(data),scale_max=np.max(data))
        dpg.configure_item("validation_mic_legend",min_scale=np.min(data),max_scale=np.max(data))
    else:
        dpg.add_heat_series(data,cols=data.shape[1],rows=data.shape[0], parent="validation_mic_y_axis",scale_min=np.min(data),scale_max=np.max(data),format="",tag="validation_mic_plot") 
        dpg.bind_colormap("validation_mic_heat", dpg.mvPlotColormap_Plasma)
        dpg.configure_item("validation_mic_legend",min_scale=np.min(data),max_scale=np.max(data))

    dpg.set_item_width("validation_mic_heat",int( dpg.get_item_height("validation_mic_heat") * (data.shape[1]/data.shape[0])))


    if dpg.does_item_exist("validation_mic_plot2"):
        dpg.set_value("validation_mic_plot2", [data]) 
        dpg.configure_item("validation_mic_plot2",cols=data.shape[1],rows=data.shape[0],scale_min=np.min(data),scale_max=np.max(data))
        dpg.configure_item("validation_mic_legend2",min_scale=np.min(data),max_scale=np.max(data))
    else:
        dpg.add_heat_series(data,cols=data.shape[1],rows=data.shape[0], parent="validation_mic_y_axis2",scale_min=np.min(data),scale_max=np.max(data),format="",tag="validation_mic_plot2")
        dpg.bind_colormap("validation_mic_heat2", dpg.mvPlotColormap_Plasma)
        dpg.configure_item("validation_mic_legend2",min_scale=np.min(data),max_scale=np.max(data))

    dpg.set_item_width("validation_mic_heat2",int( dpg.get_item_height("validation_mic_heat2") * (data.shape[1]/data.shape[0])))


    data = gradient_pressure - data_holder.mic_values[index]

    error_bound = np.max(np.abs(data))

    if dpg.does_item_exist("validation_error_plot"):
        dpg.set_value("validation_error_plot", [data]) 
        dpg.configure_item("validation_error_plot",cols=data.shape[1],rows=data.shape[0],scale_min=-error_bound,scale_max=error_bound)
        dpg.configure_item("validation_error_legend",min_scale=-error_bound,max_scale=error_bound)
    else:
        dpg.add_heat_series(data,cols=data.shape[1],rows=data.shape[0], parent="validation_error_y_axis",scale_min=-error_bound,scale_max=error_bound,format="",tag="validation_error_plot")
        dpg.bind_colormap("validation_error_heat", dpg.mvPlotColormap_RdBu)
        dpg.configure_item("validation_error_legend",min_scale=-error_bound,max_scale=error_bound)

    dpg.set_item_width("validation_error_heat",int( dpg.get_item_height("validation_error_heat") * (data.shape[1]/data.shape[0])))


    data = saturation_pressure - data_holder.mic_values[index]

    error_bound = np.max(np.abs(data))

    if dpg.does_item_exist("validation_error_plot2"):
        dpg.set_value("validation_error_plot2", [data]) 
        dpg.configure_item("validation_error_plot2",cols=data.shape[1],rows=data.shape[0],scale_min=-error_bound,scale_max=error_bound)
        dpg.configure_item("validation_error_legend2",min_scale=-error_bound,max_scale=error_bound)
    else:
        dpg.add_heat_series(data,cols=data.shape[1],rows=data.shape[0], parent="validation_error_y_axis2",scale_min=-error_bound,scale_max=error_bound,format="",tag="validation_error_plot2")
        dpg.bind_colormap("validation_error_heat2", dpg.mvPlotColormap_RdBu)
        dpg.configure_item("validation_error_legend2",min_scale=-error_bound,max_scale=error_bound)

    dpg.set_item_width("validation_error_heat2",int( dpg.get_item_height("validation_error_heat2") * (data.shape[1]/data.shape[0])))

    dpg.show_item("view_data_popup")

def update_validation_saturation_model(sender, app_data, user_data):
    global validation_selected_index
    saturations = data_holder.saturations[validation_selected_index]
    saturations[saturations <= 0] = 0.0000000000000001

    # convert gradients to pressure

    if dpg.get_value("saturation_model_selector") ==  "Physical Model":
        data = data_holder.fit_saturation_coeffs["quadratic_emission_streaming"](saturations) #TODO: IMPLEMENT THIS
    else:
        data = data_holder.fit_saturation_coeffs["quadratic_emission_streaming"](saturations) 

    saturation_pressure = data

    if dpg.does_item_exist("validation_sat_plot"):
        dpg.set_value("validation_sat_plot", [data])
        dpg.configure_item("validation_sat_plot",cols=data.shape[1],rows=data.shape[0],scale_min=np.min(data),scale_max=np.max(data))
        dpg.configure_item("validation_sat_legend",min_scale=np.min(data),max_scale=np.max(data))
    else:
        dpg.add_heat_series(data,cols=data.shape[1],rows=data.shape[0], parent="validation_sat_y_axis",scale_min=np.min(data),scale_max=np.max(data),format="",tag="validation_sat_plot")
        dpg.bind_colormap("validation_sat_heat", dpg.mvPlotColormap_Plasma)
        dpg.configure_item("validation_sat_legend",min_scale=np.min(data),max_scale=np.max(data))

    data = saturation_pressure - data_holder.mic_values[validation_selected_index]

    error_bound = np.max(np.abs(data))

    if dpg.does_item_exist("validation_error_plot2"):
        dpg.set_value("validation_error_plot2", [data])
        dpg.configure_item("validation_error_plot2",cols=data.shape[1],rows=data.shape[0],scale_min=-error_bound,scale_max=error_bound)
        dpg.configure_item("validation_error_legend2",min_scale=-error_bound,max_scale=error_bound)
    else:
        dpg.add_heat_series(data,cols=data.shape[1],rows=data.shape[0], parent="validation_error_y_axis2",scale_min=-error_bound,scale_max=error_bound,format="",tag="validation_error_plot2")
        dpg.bind_colormap("validation_error_heat2", dpg.mvPlotColormap_RdBu)
        dpg.configure_item("validation_error_legend2",min_scale=-error_bound,max_scale=error_bound)

def update_validation_gradient_model(sender, app_data, user_data): 
    global validation_selected_index
    gradients = data_holder.gradients[validation_selected_index]
    gradients[gradients <= 0] = 0.0000000000000001

    # convert gradients to pressure

    if dpg.get_value("gradient_model_selector") ==  "Physical Model":
        data = press_from_mesh(gradients)
    else:
        data = press_from_grad_fit(gradients, data_holder.fit_grad_coeffs[1], data_holder.fit_grad_coeffs[2])

    gradient_pressure = data

    if dpg.does_item_exist("validation_grad_plot"):
        dpg.set_value("validation_grad_plot", [data]) 
        dpg.configure_item("validation_grad_plot",cols=data.shape[1],rows=data.shape[0],scale_min=np.min(data),scale_max=np.max(data))
        dpg.configure_item("validation_grad_legend",min_scale=np.min(data),max_scale=np.max(data))
    else:
        dpg.add_heat_series(data,cols=data.shape[1],rows=data.shape[0], parent="validation_grad_y_axis",scale_min=np.min(data),scale_max=np.max(data),format="",tag="validation_grad_plot")
        dpg.bind_colormap("validation_grad_heat", dpg.mvPlotColormap_Plasma)
        dpg.configure_item("validation_grad_legend",min_scale=np.min(data),max_scale=np.max(data))

    data = gradient_pressure - data_holder.mic_values[validation_selected_index]

    error_bound = np.max(np.abs(data))

    if dpg.does_item_exist("validation_error_plot"):
        dpg.set_value("validation_error_plot", [data]) 
        dpg.configure_item("validation_error_plot",cols=data.shape[1],rows=data.shape[0],scale_min=-error_bound,scale_max=error_bound)
        dpg.configure_item("validation_error_legend",min_scale=-error_bound,max_scale=error_bound)
    else:
        dpg.add_heat_series(data,cols=data.shape[1],rows=data.shape[0], parent="validation_error_y_axis",scale_min=-error_bound,scale_max=error_bound,format="",tag="validation_error_plot")
        dpg.bind_colormap("validation_error_heat", dpg.mvPlotColormap_RdBu)
        dpg.configure_item("validation_error_legend",min_scale=-error_bound,max_scale=error_bound)


def update_validation_thermal_frame(sender, app_data, user_data):
    # update the frame viewed

    global validation_selected_index

    selected_frame = int(dpg.get_value("validation_time_slider"))

    data = data_holder.thermals_g_nr[validation_selected_index][selected_frame]

    dpg.set_value("validation_data_plot", [data]) 
    dpg.configure_item("validation_data_plot",cols=data.shape[1],rows=data.shape[0],scale_min=np.min(data),scale_max=np.max(data))
    dpg.configure_item("validation_data_legend",min_scale=np.min(data),max_scale=np.max(data))




def convert_measurement_data(sender, app_data, user_data):
    index = dpg.get_item_configuration("list_box_measurement")["items"].index(dpg.get_value("list_box_measurement"))

    #TODO: choose the conversion method

    dpg.show_item("load_data_popup2")

    data_holder_measurement.pressures = []

    dpg.set_value("load_data_loading_bar2", 0)
    dpg.configure_item("load_data_loading_bar2", overlay=f"{0}%")

    data_holder_measurement.gradients = []
    data_holder_measurement.gradients_temp = []
    data_holder_measurement.gradients_maxs = []

    data_holder_measurement.pressures = []

    # convert data here

    dpg.set_value(loading_files_text2, f"Converting file {1} out of {1}")

    cutoffs = [1,5,12.5,15,25]
    samples = [10,9,8,6,5,4]

    grad, temp = extract_gradient_and_temp_timestamped_NUTS(data_holder_measurement.thermals_nr[index], data_holder_measurement.timestamps[index], data_holder_measurement.start_indices[index], cutoffs=cutoffs, samples=samples)

    grad[grad<=0] = 0.0000000000000001 #TODO: fix this?

    data_holder_measurement.gradients[index] = grad
    data_holder_measurement.gradients_temp[index] = temp
    data_holder_measurement.gradients_maxs[index] = np.max(grad)

    data_holder_measurement.pressures[index] = press_from_mesh(data_holder_measurement.gradients[index])

    data_holder_measurement.converted_indices[index] = index

    dpg.set_value("load_data_loading_bar2", 1)

    dpg.configure_item("load_data_loading_bar2", overlay=f"{100}%")
    
    new_files = data_holder_measurement.name

    for index, file_name in enumerate(new_files):
        if index not in data_holder_measurement.converted_indices:
            new_files[index] = "UNCONVERTED "+file_name

    dpg.hide_item("load_data_popup2")

    dpg.configure_item("list_box_measurement_converted",items=new_files)

    #data_holder_measurement
    dpg.enable_item("view_converted_data_button")


def convert_all_measurement_data(sender, app_data, user_data):

    #TODO: choose the conversion method

    dpg.show_item("load_data_popup2")

    data_holder_measurement.pressures = []

    dpg.set_value("load_data_loading_bar2", 0)
    dpg.configure_item("load_data_loading_bar2", overlay=f"{0}%")

    data_holder_measurement.gradients = []
    data_holder_measurement.gradients_temp = []
    data_holder_measurement.gradients_maxs = []

    data_holder_measurement.pressures = []


    for index, _ in enumerate(data_holder_measurement.timestamps):
        # convert data here

        dpg.set_value(loading_files_text2, f"Converting file {index + 1} out of {len(data_holder_measurement.timestamps)}")

        cutoffs = [1,5,12.5,15,25]
        samples = [10,9,8,6,5,4]

        grad, temp = extract_gradient_and_temp_timestamped_NUTS(data_holder_measurement.thermals_nr[index], data_holder_measurement.timestamps[index], data_holder_measurement.start_indices[index], cutoffs=cutoffs, samples=samples)

        grad[grad<=0] = 0.0000000000000001 #TODO: fix this?

        data_holder_measurement.gradients[index] = grad
        data_holder_measurement.gradients_temp[index] = temp
        data_holder_measurement.gradients_maxs[index] = np.max(grad)
        data_holder_measurement.converted_indices[index] = index

        data_holder_measurement.pressures[index] = press_from_mesh(data_holder_measurement.gradients[index])

        dpg.set_value("load_data_loading_bar2", (index + 1 )/len(data_holder_measurement.timestamps))

        dpg.configure_item("load_data_loading_bar2", overlay=f"{ int(((index + 1)/len(data_holder_measurement.timestamps))*100) }%")
    
    new_files = data_holder_measurement.name

    dpg.hide_item("load_data_popup2")

    dpg.configure_item("list_box_measurement_converted",items=new_files)

    #data_holder_measurement
    dpg.enable_item("view_converted_data_button")

def load_thermal_files(files):

    dpg.show_item("load_data_popup2")

    dpg.set_value("load_data_loading_bar2", 0)
    dpg.configure_item("load_data_loading_bar2", overlay=f"{0}%")

    for index, file in enumerate(files):

        dpg.set_value(loading_files_text2, f"Loading file {index + 1} out of {len(files)}")

        # get data from file

        #TODO: allow selection of NUC region

        #TODO: Allow specification of start/end timestamps of stimuli and x and y ROI

        timestamps, start, ambient, thermal, thermal_nr = load_thermal_file(file)

        # put data into storage

        data_holder_measurement.timestamps.append(timestamps)
        data_holder_measurement.start_indices.append(start)
        data_holder_measurement.ambients.append(ambient)
        data_holder_measurement.thermals.append(thermal)
        data_holder_measurement.thermals_nr.append(thermal_nr)
        data_holder_measurement.name.append(file)

        data_holder_measurement.gradients.append(None)
        data_holder_measurement.gradients_maxs.append(None)
        data_holder_measurement.gradients_temp.append(None)
        data_holder_measurement.saturations.append(None)
        data_holder_measurement.saturations_maxs.append(None)
        data_holder_measurement.pressures.append(None)
        data_holder_measurement.converted_indices.append(None)

        dpg.set_value("load_data_loading_bar2", (index + 1 )/len(files))

        dpg.configure_item("load_data_loading_bar2", overlay=f"{ int(((index + 1)/len(files))*100) }%")


    dpg.hide_item("load_data_popup2")

    dpg.enable_item("convert_data_button")
    dpg.enable_item("convert_all_data_button")

    new_files = dpg.get_item_configuration("list_box_measurement")['items'] + files
    dpg.configure_item("list_box_measurement",items=new_files)




def load_data(sender, app_data, user_data):

    dpg.show_item("load_data_popup")

    data_holder.timestamps = []
    data_holder.start_indices = []
    data_holder.mic_values = []
    data_holder.ambients = []
    data_holder.thermals_g = []
    data_holder.thermals_g_nr = []
    data_holder.thermals_s = []
    data_holder.thermals_s_nr = []
    data_holder.group_name = []
    data_holder.name = []


    tabs = dpg.get_item_children("thermal_data_tab_bar")[1][:-1]
    tab_total = len(tabs)
    tab_count = 0
    file_count = 0
    final_count = 0

    file_total = 0

    for tab in tabs:
        file_total += len(dpg.get_item_configuration(str(tab)+"_list_box_therm")['items'])

    for tab in tabs:
        tab_count += 1
        #print(f"Group: {tab}")
        #print(dpg.get_item_configuration(str(tab)+"_list_box_therm")['items'])
        file_count += len(dpg.get_item_configuration(str(tab)+"_list_box_therm")['items'])

        for index, therm_file in enumerate(dpg.get_item_configuration(str(tab)+"_list_box_therm")['items']):
            mic_file = dpg.get_item_configuration(str(tab)+"_list_box_mic")['items'][index]
            print(f"Extracting file {index + 1 + final_count} out of {file_total}. Therm: {therm_file}, mic: {mic_file}")

            dpg.set_value(loading_files_text, f"Loading file {index + 1 + final_count} out of {file_total}")

            xs = 100
            ys = 100

            xs = [[58, 169]]
            size = get_size_from_mic_file(mic_file)
            ratio = size[1]/size[0]
            heights = [ x[1] - x[0] for x in xs]
            widths = [height*ratio for height in heights]

            start_ys = [90]

            ys = [ [y,y+int(widths[index])] for index, y in enumerate(start_ys)]

            if ratio != 1:
                timestamps, start_index, mic_values, ambient, downscaled_thermals_s, downscaled_thermals_g, downscaled_thermals_s_nr, downscaled_thermals_g_nr   = load_thermal_and_mic_file(therm_file,mic_file,xs[0],ys[0],9,25,15)
            else:
                timestamps, start_index, mic_values, ambient, downscaled_thermals_s, downscaled_thermals_g, downscaled_thermals_s_nr, downscaled_thermals_g_nr   = load_thermal_and_mic_file(therm_file,mic_file,xs[0],ys[0])

            data_holder.timestamps.append(timestamps)
            data_holder.start_indices.append(start_index)
            data_holder.mic_values.append(mic_values)
            data_holder.ambients.append(ambient)
            data_holder.thermals_g.append(downscaled_thermals_g)
            data_holder.thermals_g_nr.append(downscaled_thermals_g_nr)
            data_holder.thermals_s.append(downscaled_thermals_s)
            data_holder.thermals_s_nr.append(downscaled_thermals_s_nr)
            data_holder.group_name.append(dpg.get_item_configuration(tab)['label'])
            data_holder.name.append((therm_file,mic_file))

            dpg.set_value("load_data_loading_bar", (index + 1 + final_count)/file_total)

            dpg.configure_item("load_data_loading_bar", overlay=f"{ int(((index + 1 + final_count)/file_total)*100) }%")

        
        final_count += file_count

        #print(dpg.get_item_configuration(str(tab)+"_list_box_mic")['items'])

    y_peak = []

    for v in data_holder.mic_values:
        y_peak.append(np.max(v))

    data_holder.mic_peaks = y_peak

    dpg.hide_item("load_data_popup")

    dpg.set_value(Data_text, f"{tab_count} data groups loaded comprising of {file_count} files")

    if file_count > 0 :
        dpg.configure_item(extract_gradient_button, enabled=True)


def extract_gradients_and_saturations(sender, app_data, user_data):

    cutoffs = [1,5,12.5,15,25]
    samples = [10,9,8,6,5,4]

    group_indices = {}

    for index, group_name in enumerate(data_holder.group_name):
        group_index_list = group_indices.get(group_name, [])
        group_index_list.append(index)
        group_indices[group_name] = group_index_list

    for index, thermal in enumerate(data_holder.thermals_g):
        grad, temp = extract_gradient_and_temp_timestamped_NUTS(data_holder.thermals_g_nr[index], data_holder.timestamps[index], data_holder.start_indices[index], cutoffs=cutoffs, samples=samples)
        data_holder.gradients.append(grad)
        data_holder.gradients_temp.append(temp)
        data_holder.gradients_maxs.append(np.max(grad))

    
    data_points = 0

    for group_name in group_indices:

        mic_vals = [data_holder.mic_values[i] for i in group_indices[group_name]]
        grads = [data_holder.gradients[i] for i in group_indices[group_name]]

        x = grads[0].flatten()
        y = mic_vals[0].flatten()

        for ind,_ in enumerate(mic_vals):
            if ind != 0:
                x = np.concatenate((x,grads[ind].flatten()))
                y = np.concatenate((y,mic_vals[ind].flatten()))

        data_points += len(x)

        dpg.add_scatter_series(x,y,label=f"{group_name} Measurements", parent="gradient_y_axis")


    dpg.add_scatter_series(data_holder.gradients_maxs,data_holder.mic_peaks,label=f"Peak Measurements", parent="gradient_y_axis")
    dpg.fit_axis_data("gradient_y_axis")
    dpg.fit_axis_data("gradient_x_axis")

    for index, thermal in enumerate(data_holder.thermals_g):
        sats = extract_non_max_steady_state_from_diff(data_holder.thermals_s_nr[index])
        data_holder.saturations.append(sats)
        data_holder.saturations_maxs.append(np.max(sats))

    data_points = 0

    for group_name in group_indices:

        mic_vals = [data_holder.mic_values[i] for i in group_indices[group_name]]
        sats = [data_holder.saturations[i] for i in group_indices[group_name]]

        x = sats[0].flatten()
        y = mic_vals[0].flatten()

        for ind,_ in enumerate(mic_vals):
            if ind != 0:
                x = np.concatenate((x,sats[ind].flatten()))
                y = np.concatenate((y,mic_vals[ind].flatten()))

        data_points += len(x)

        dpg.add_scatter_series(x,y,label=f"{group_name} Measurements",parent="saturation_y_axis")


    dpg.add_scatter_series(data_holder.saturations_maxs,data_holder.mic_peaks,label=f"Peak Measurements", parent="saturation_y_axis")

    dpg.fit_axis_data("saturation_y_axis")
    dpg.fit_axis_data("saturation_x_axis")

    dpg.set_value(Gradient_text, f"{data_points} data points loaded")
    if data_points > 0:
        dpg.configure_item(fit_gradient_button, enabled=True)
        for child in dpg.get_item_children("thermal_data_tab_bar2")[1]:
            dpg.enable_item(str(child)+"view_converted_data_button_validation")



def calculate_attenuation(sender, app_data, user_data):
    
    frequency = dpg.get_value("sound_frequency")

    air_conductivity = dpg.get_value("thermal_conductivity_air")
    air_heat_capacity = dpg.get_value("specific_heat_capacity_air")
    dynamic_viscosity_air = dpg.get_value("dynamic_viscosity_air")
    air_density = dpg.get_value("density_air")

    width = dpg.get_value("hole_radius_nylon")*2 # Pore size
    air_speed = dpg.get_value("speed_of_sound_air")
    adiabatic_index = dpg.get_value("adiabatic_index_air")

    attenuation =  calculate_attenuation_mesh(frequency,air_density,dynamic_viscosity_air,air_heat_capacity,air_conductivity,width,air_speed,adiabatic_index)

    dpg.set_value(attenuation_text, f"{attenuation}")

    try:
        absorbtion = float(dpg.get_value(absorbtion_text))
        dpg.configure_item(plot_gradient_physical_model_button, enabled=True)
    except:
        pass


def calculate_absorbtion(sender, app_data, user_data):

    frequency = dpg.get_value("sound_frequency")
    dynamic_viscosity_air = dpg.get_value("dynamic_viscosity_air")
    air_speed = dpg.get_value("speed_of_sound_air")
    air_density = dpg.get_value("density_air")
    nylon_depth = dpg.get_value("mesh_thickness_nylon")
    width = dpg.get_value("hole_radius_nylon")*2 # Pore size

    porosity = dpg.get_value("porosity_nylon") #19%


    absorbtion = calculate_absorbtion_mesh(nylon_depth, width, porosity, air_density, dynamic_viscosity_air, frequency, air_speed)
    
    dpg.set_value(absorbtion_text, f"{absorbtion}")

    try:
        attenuation = float(dpg.get_value(attenuation_text))
        dpg.configure_item(plot_gradient_physical_model_button, enabled=True)
    except:
        pass

def press_from_mesh(gradient):

    air_speed = dpg.get_value("speed_of_sound_air")
    air_density = dpg.get_value("density_air")

    nylon_density = dpg.get_value("density_nylon")
    nylon_heat_capacity = dpg.get_value("specific_heat_capacity_nylon")
    nylon_thread_radius = dpg.get_value("thread_radius_nylon")
    nylon_depth = dpg.get_value("mesh_thickness_nylon")

    attenuation = float(dpg.get_value(attenuation_text))
    absorbtion = float(dpg.get_value(absorbtion_text))

    pixel_size = 1e-3

    return press_from_gradient_mesh(gradient, attenuation, absorbtion,nylon_thread_radius, pixel_size, nylon_depth, nylon_heat_capacity, nylon_density, air_density, air_speed)


def plot_mesh_gradient(sender, app_data, user_data):

    x = []
    y = []

    for grad in np.linspace(0,50,200):
        x.append(grad)
        y.append(press_from_mesh(grad))

    if dpg.does_item_exist("physical_model_plot"):
        dpg.set_value("physical_model_plot", [x,y])
    else:
        dpg.add_line_series(x,y,label=f"Physical Model",parent="gradient_y_axis", tag="physical_model_plot")

def fit_gradient_and_saturation(sender, app_data, user_data):
    # ok time to fit gradient
    gradients_calib_sorted = [x for _, x in sorted(zip(data_holder.mic_peaks, data_holder.gradients_maxs), key=lambda pair: pair[0])]
    pressures_calib_sorted = sorted(data_holder.mic_peaks)
    gradients_calib_sorted = np.array(gradients_calib_sorted)
    pressures_calib_sorted = np.array(pressures_calib_sorted)


    att_coef, a5, b5 = calibrate_gradient(gradients_calib_sorted, pressures_calib_sorted, print_report=False)

    data_holder.fit_grad_coeffs = [att_coef, a5, b5]

    grad = np.linspace(0,50,200)
    x = grad.tolist()
    y = press_from_grad_fit(grad,a5,b5).tolist()

    if dpg.does_item_exist("fit_gradient_plot"):
        dpg.set_value("fit_gradient_plot", [x,y])
    else:
        dpg.add_line_series(x,y,label=f"Fit Square Root Model",parent="gradient_y_axis", tag="fit_gradient_plot")

    steady_state_increases_calib_sorted = [x for _, x in sorted(zip(data_holder.mic_peaks, data_holder.saturations_maxs), key=lambda pair: pair[0])]
    steady_state_increases_calib_sorted = np.array(steady_state_increases_calib_sorted)


    thermal_to_pressure_naive, thermal_to_pressure_quadratic, thermal_to_pressure_quadratic_true, thermal_to_pressure_quadratic_true_emission,thermal_to_pressure_quadratic_true_emission_variable_h, thermal_to_pressure_quadratic_true_variable_h  = calibrate_steady_state_naive(steady_state_increases_calib_sorted, pressures_calib_sorted,print_report=False,streaming_cutoff=2.8)

    data_holder.fit_saturation_coeffs = {"naive": thermal_to_pressure_naive, "quadratic": thermal_to_pressure_quadratic, "quadratic-true": thermal_to_pressure_quadratic_true, "quadratic_emission": thermal_to_pressure_quadratic_true_emission,"quadratic_emission_streaming":thermal_to_pressure_quadratic_true_emission_variable_h, "quadratic_streaming": thermal_to_pressure_quadratic_true_variable_h}

    sat = np.linspace(0.00000001,50,200)
    x = sat.tolist()
    y = thermal_to_pressure_quadratic_true_emission_variable_h(sat).tolist()

    if dpg.does_item_exist("quad_emiss_stream_plot"):
        dpg.set_value("quad_emiss_stream_plot", [x,y])
    else:
        dpg.add_line_series(x,y,label=f"quadratic w/ emission & streaming",parent="saturation_y_axis",tag="quad_emiss_stream_plot")


    sat = np.linspace(0.00000001,50,200)
    x = sat.tolist()
    y = thermal_to_pressure_quadratic_true_emission(sat).tolist()

    if dpg.does_item_exist("quad_emiss_plot"):
        dpg.set_value("quad_emiss_plot", [x,y])
    else:
        dpg.add_line_series(x,y,label=f"quadratic w/ emission",parent="saturation_y_axis",tag="quad_emiss_plot")


    sat = np.linspace(0.00000001,50,200)
    x = sat.tolist()
    y = thermal_to_pressure_quadratic_true_variable_h(sat).tolist()

    if dpg.does_item_exist("quad_stream_plot"):
        dpg.set_value("quad_stream_plot", [x,y])
    else:
        dpg.add_line_series(x,y,label=f"quadratic w/ streaming",parent="saturation_y_axis",tag="quad_stream_plot")


    sat = np.linspace(0.00000001,50,200)
    x = sat.tolist()
    y = thermal_to_pressure_quadratic_true(sat).tolist()

    if dpg.does_item_exist("quad_plot"):
        dpg.set_value("quad_plot", [x,y])
    else:
        dpg.add_line_series(x,y,label=f"quadratic",parent="saturation_y_axis",tag="quad_plot")

def remove_data_group(sender, app_data, user_data):

    children = dpg.get_item_children("thermal_data_tab_bar2")

    for child in children[1]:
        if dpg.get_item_configuration(child)["label"] == dpg.get_item_configuration(dpg.get_item_info(sender)["parent"])["label"]:
            dpg.delete_item(child)

    del tabs[tabs.index(dpg.get_item_configuration(dpg.get_item_info(sender)["parent"])["label"])]

    dpg.delete_item(dpg.get_item_info(sender)["parent"])


def rename_data_group(sender, app_data, user_data):
    

    children = dpg.get_item_children("thermal_data_tab_bar2")

    for child in children[1]:
        if dpg.get_item_configuration(child)["label"] == dpg.get_item_configuration(dpg.get_item_info(sender)["parent"])["label"]:
            dpg.configure_item(child,label=app_data)

    tabs[tabs.index(dpg.get_item_configuration(dpg.get_item_info(sender)["parent"])["label"])] = app_data

    dpg.configure_item(dpg.get_item_info(sender)["parent"],label=app_data)

def add_thermal_files(sender, app_data, user_data):

    dpg.configure_item(user_data[0],items=list(app_data["selections"].values()))

    children = dpg.get_item_children("thermal_data_tab_bar2")

    for child in children[1]:
        if dpg.get_item_configuration(child)["label"] == dpg.get_item_configuration(int(user_data[1]))["label"]:
            tab = child
            break

    dpg.configure_item(str(tab)+"_list_box_therm2",items=list(app_data["selections"].values()))
                            

def add_mic_files(sender, app_data, user_data):
    dpg.configure_item(user_data[0],items=list(app_data["selections"].values()))

    children = dpg.get_item_children("thermal_data_tab_bar2")

    for child in children[1]:
        if dpg.get_item_configuration(child)["label"] == dpg.get_item_configuration(int(user_data[1]))["label"]:
            tab = child
            break

    dpg.configure_item(str(tab)+"_list_box_mic2",items=list(app_data["selections"].values()))

dpg.colormap_registry()

#camera = flircamera.SingleCamera(config='highspeed.json', share=False)
#camera.begin_acquisition('test.avi', record_kws=dict(bitrate=1e6))
#for k in range(200*60*5):
#    im = camera.get_frame()
#camera.end_acquisition()
#camera.finalize()

def connect_thermal_camera(sender, app_data, user_data):
    global camera
    camera = flircamera.CameraManager()
    camera.get_camera()
    camera.setup_camera()
    dpg.enable_item("live_convert_button")
    dpg.disable_item("connect_camera_button")
    dpg.enable_item("disconnect_camera_button")
    dpg.set_value("camera_status_text", "Camera Connected")

def disconnect_thermal_camera(sender, app_data, user_data):
    global camera
    global recording
    camera.release_camera(acquisition_status=True)
    recording = False
    dpg.disable_item("live_convert_button")
    dpg.enable_item("connect_camera_button")
    dpg.disable_item("disconnect_camera_button")
    dpg.set_value("camera_status_text", "No Camera Connected")

def live_camera_capture(sender, app_data, user_data):
    global camera
    global recording
    camera.begin_acquisition()
    recording = True

    frame, frame_status = camera.capture_frame()

    print(frame_status)

    data = frame

    if dpg.does_item_exist("camera_data_plot"):
        dpg.set_value("camera_data_plot", [data]) 
        dpg.configure_item("camera_data_plot",cols=data.shape[1],rows=data.shape[0],scale_min=np.min(data),scale_max=np.max(data))
        dpg.configure_item("camera_data_legend",min_scale=np.min(data),max_scale=np.max(data))
    else:
        dpg.add_heat_series(data,cols=data.shape[1],rows=data.shape[0], parent="camera_y_axis",scale_min=np.min(data),scale_max=np.max(data),format="",tag="camera_data_plot")
        dpg.bind_colormap("camera_data_heat", dpg.mvPlotColormap_Plasma)
        dpg.configure_item("camera_data_legend",min_scale=np.min(data),max_scale=np.max(data))

    dpg.set_item_width("camera_data_heat",int( dpg.get_item_height("camera_data_heat") * (data.shape[1]/data.shape[0])))

    dpg.show_item("view_camera_popup")


def camera_hovered(sender, app_data, user_data):
    mouse_pos = dpg.get_plot_mouse_pos()
    # mouse pos is from 0 to 1 in both axis, 1,1 is top right, 0,0 is bottom left
    data = dpg.get_value("camera_data_plot")
    #data = np.array(dpg.get_value("camera_data_plot"))
    plot_config = dpg.get_item_configuration("camera_data_plot")

    height = plot_config["rows"]
    width = plot_config["cols"]


    temperature = data[0][int((1-mouse_pos[1])*height)*width + int(mouse_pos[0]*width)]
    dpg.set_value("camera_tooltip",f"{temperature:.2f}")
    

with dpg.window(label="Example Window", tag="Primary Window"):
    with dpg.tab_bar():
        with dpg.tab(label="Measurement"):
            with dpg.window(tag="load_data_popup2",modal=False,show=False,width=600,height=110,pos=[100,100]):
                with dpg.table(header_row=False, resizable=False, policy=dpg.mvTable_SizingStretchSame,
                    borders_outerH=False,borders_innerH=False, borders_innerV=False, borders_outerV=False):

                    dpg.add_table_column(label="Header 1",no_resize=True,no_reorder=True,no_hide=True,width_stretch=True)
                    dpg.add_table_column(label="Header 2",no_resize=True,no_reorder=True,no_hide=True,width_stretch=True)
                    dpg.add_table_column(label="Header 3",no_resize=True,no_reorder=True,no_hide=True,width_stretch=True)

                    with dpg.table_row():
                        for j in range(0, 3):
                            with dpg.table_cell():
                                if j == 1:
                                    loading_files_text2 = dpg.add_text("Loading files!")

                
                dpg.add_separator()
                with dpg.group(horizontal=True):
                    dpg.add_progress_bar(tag="load_data_loading_bar2",label="Progress Bar", default_value=0, overlay="0%",height=25,width=550)
                    dpg.add_loading_indicator(style=1,radius=1.25)#1.25

            with dpg.window(tag="view_camera_popup",modal=False,show=False,width=1400,height=900,pos=[0,0],label="Camera Live Feed"):
                with dpg.group(horizontal=True):
                    with dpg.plot(label="Raw Thermal Video", tag="camera_data_heat",height=400,width=800,anti_aliased=False):
                        #dpg.add_plot_legend(location=dpg.mvPlot_Location_SouthEast)
                        dpg.add_plot_axis(dpg.mvXAxis, tag="camera_x_axis",no_tick_labels=False)
                        dpg.add_plot_axis(dpg.mvYAxis, tag="camera_y_axis",no_tick_labels=False)
                    dpg.add_colormap_scale(min_scale=0,max_scale=1,tag="camera_data_legend",height=300,colormap=dpg.mvPlotColormap_Plasma)
                    with dpg.tooltip(parent="camera_y_axis"):
                        dpg.add_text("A tooltip",tag="camera_tooltip")
                    with dpg.item_handler_registry(tag="widget_handler"):
                        dpg.add_item_hover_handler(callback=camera_hovered)

                    with dpg.plot(label="Converted Thermal Video", tag="camera_conv_heat",height=400,width=800,anti_aliased=False):
                        #dpg.add_plot_legend(location=dpg.mvPlot_Location_SouthEast)
                        dpg.add_plot_axis(dpg.mvXAxis, tag="camera_conv_x_axis",no_tick_labels=False)
                        dpg.add_plot_axis(dpg.mvYAxis, tag="camera_conv_y_axis",no_tick_labels=False)
                    dpg.add_colormap_scale(min_scale=0,max_scale=1,tag="camera_conv_legend",height=300,colormap=dpg.mvPlotColormap_Plasma)

                    dpg.bind_item_handler_registry("camera_data_heat", "widget_handler")


            dpg.add_radio_button(("Gradient", "Steady State", "Steady State Physical Model"), horizontal=True)

            calibration_text = dpg.add_text("No Calibration curve currently set")

            with dpg.group(horizontal=True):
               load_calib_file = dpg.add_button(label="Load Calibration File")
               load_calib_from_valid = dpg.add_button(label="Use Calibration From Validation/Training Tab", callback=dpg.set_value(calibration_text, "Using Calibration From Training Tab"))

            with dpg.group(width=200):
                dpg.add_input_double(label="Sound Frequency", max_value=200000.0, format="%.0f Hz", default_value=40000.0, step=1000)

            camera_status_text = dpg.add_text("No Camera Connected",tag="camera_status_text")

            with dpg.file_dialog(label="Choose Thermal Files", width=1000, height=800, show=False, callback=lambda s, a, u : load_thermal_files(list(a["selections"].values()))) as fd3:
                dpg.add_file_extension(".seq",color=(0,153,0, 255))
                dpg.add_file_extension("", custom_text="[Directory]", color=(255, 150, 150, 255))

            with dpg.group(horizontal=True):
                dpg.add_button(label="Connect Camera",callback=connect_thermal_camera,tag="connect_camera_button")
                dpg.add_button(label="Disconnect Camera",callback=disconnect_thermal_camera,enabled=False,tag="disconnect_camera_button")
                dpg.add_button(label="Record Data",enabled=False)
                dpg.add_button(label="Load Data From File",user_data=fd3, callback=lambda s, a, u: dpg.configure_item(u, show=True))
            dpg.add_text("Data currently recorded/loaded:")
            measurement_files = dpg.add_listbox(tag="list_box_measurement")
            with dpg.group(horizontal=True):
                dpg.add_button(label="Convert All Data", enabled=False, tag="convert_all_data_button", callback=convert_all_measurement_data)
                dpg.add_button(label="Convert Selected Data", enabled=False, tag="convert_data_button", callback=convert_measurement_data)
                dpg.add_button(label="Live-Convert from Camera using steady state",enabled=False,tag="live_convert_button",callback=live_camera_capture)
            dpg.add_text("Data Currently Converted:")
            dpg.add_listbox(tag="list_box_measurement_converted")
            dpg.add_button(label="View Selected Data",enabled=False, tag="view_converted_data_button",callback=view_converted_measurement)

            with dpg.group(horizontal=True):
                with dpg.plot(label="Converted Data", tag="converted_data_heat",height=800,width=800,anti_aliased=False):
                    #dpg.add_plot_legend(location=dpg.mvPlot_Location_SouthEast)
                    dpg.add_plot_axis(dpg.mvXAxis, tag="connverted_x_axis",no_tick_labels=False)
                    dpg.add_plot_axis(dpg.mvYAxis, tag="connverted_y_axis",no_tick_labels=False)
                dpg.add_colormap_scale(min_scale=0,max_scale=1,tag="converted_data_legend",height=800,colormap=dpg.mvPlotColormap_Plasma)






        with dpg.tab(label="Training", tag="loading_tab"):

            with dpg.collapsing_header(label="Select and Group Data Files"):
                with dpg.group(indent=10):
                    with dpg.tab_bar(tag="thermal_data_tab_bar") as tab_bar:
                        with dpg.file_dialog(label="Choose Thermal Files", width=1000, height=800, show=False, callback=add_thermal_files) as fd:
                        #with dpg.file_dialog(label="Choose Thermal Files", width=300, height=400, show=False, callback=lambda s, a, u : print(a), tag="training_file_dialog") as fd:
                            dpg.add_file_extension(".seq",color=(0,153,0, 255))
                            dpg.add_file_extension("", custom_text="[Directory]", color=(255, 150, 150, 255))

                        with dpg.file_dialog(label="Choose Microphone Files", width=1000, height=800, show=False, callback=add_mic_files) as fd2:
                        #with dpg.file_dialog(label="Choose Thermal Files", width=300, height=400, show=False, callback=lambda s, a, u : print(a["selections"].values()), tag="training_file_dialog") as fd:
                            dpg.add_file_extension(".csv",color=(0,153,0, 255))
                            dpg.add_file_extension("", custom_text="[Directory]", color=(255, 150, 150, 255))

                        with dpg.tab(label="tab 1") as tab:
                            dpg.add_input_text(hint="Rename Data Group here",callback=rename_data_group, user_data=tab)
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="Choose Thermal Files", user_data=fd, callback=lambda s, a, u: dpg.configure_item(u, show=True,user_data=(str(tab)+"_list_box_therm",str(tab))))
                                dpg.add_button(label="Choose Microphone Files", user_data=fd2, callback=lambda s, a, u: dpg.configure_item(u, show=True,user_data=(str(tab)+"_list_box_mic",str(tab))))
                            with dpg.group(horizontal=True,width=600):
                                dpg.add_listbox(tag=str(tab)+"_list_box_therm")
                                dpg.add_listbox(tag=str(tab)+"_list_box_mic")
                            dpg.add_button(label="Remove Data Group", callback=remove_data_group, user_data=tab)

                        dpg.add_tab_button(label="+", callback=add_tab_thermal_data,tag="thermal_data_add_tab_button")

            Data_text = dpg.add_text("No Data loaded yet")
            dpg.add_button(label="Load Data", callback=load_data)

            with dpg.window(tag="load_data_popup",modal=False,show=False,width=500,height=110,pos=[100,100]):
                with dpg.table(header_row=False, resizable=False, policy=dpg.mvTable_SizingStretchSame,
                    borders_outerH=False,borders_innerH=False, borders_innerV=False, borders_outerV=False):

                    dpg.add_table_column(label="Header 1",no_resize=True,no_reorder=True,no_hide=True,width_stretch=True)
                    dpg.add_table_column(label="Header 2",no_resize=True,no_reorder=True,no_hide=True,width_stretch=True)
                    dpg.add_table_column(label="Header 3",no_resize=True,no_reorder=True,no_hide=True,width_stretch=True)

                    with dpg.table_row():
                        for j in range(0, 3):
                            with dpg.table_cell():
                                if j == 1:
                                    loading_files_text = dpg.add_text("Loading files!")

                
                dpg.add_separator()
                with dpg.group(horizontal=True):
                    dpg.add_progress_bar(tag="load_data_loading_bar",label="Progress Bar", default_value=0, overlay="0%",height=25,width=450)
                    dpg.add_loading_indicator(style=1,radius=1.25)#1.25

            Gradient_text = dpg.add_text("No Gradients or Saturations extracted")
            extract_gradient_button = dpg.add_button(label="Extract Gradients and Saturations", callback=extract_gradients_and_saturations,enabled=False)
            fit_gradient_button = dpg.add_button(label="Fit Models", callback=fit_gradient_and_saturation,enabled=False)

            with dpg.group(horizontal=True):
                with dpg.plot(label="gradient scatter plot", tag="gradient_scatter_plot",height=400,width=800,anti_aliased=True):
                    dpg.add_plot_legend(location=dpg.mvPlot_Location_SouthEast)
                    dpg.add_plot_axis(dpg.mvXAxis, label="Initial Temperature Gradient (K/s)", tag="gradient_x_axis")
                    dpg.add_plot_axis(dpg.mvYAxis, label="Pressure (Pa RMS)",tag="gradient_y_axis")
                    
                    #dpg.set_axis_limits("gradient_y_axis",-200,3300)
                    #dpg.set_axis_limits("gradient_x_axis",-2,38)

                with dpg.plot(label="saturation scatter plot", tag="saturation_scatter_plot",height=400,width=800,anti_aliased=True):
                    dpg.add_plot_legend(location=dpg.mvPlot_Location_SouthEast)
                    dpg.add_plot_axis(dpg.mvXAxis, label="Max Steady State Temperature Rise (K)",tag="saturation_x_axis")
                    dpg.add_plot_axis(dpg.mvYAxis, label="Pressure (Pa RMS)", tag="saturation_y_axis")

                    #dpg.set_axis_limits("saturation_y_axis",-200,3300)
                    #dpg.set_axis_limits("saturation_x_axis",-2,38)


            with dpg.collapsing_header(label="Mesh and Air Properties"):
                with dpg.group(width=200):
                    with dpg.tree_node(label="Mesh Geometry"):
                        dpg.add_input_double(tag="mesh_thickness_nylon",label="Mesh Thickness", format="%.7f m", default_value=6.5e-5, step=0.1e-5)
                        dpg.add_input_double(tag="thread_radius_nylon",label="Thread Radius", format="%.7f m", default_value=3.3e-5, step=0.1e-5)
                        dpg.add_slider_double(tag="porosity_nylon",label="Porosity", max_value=1, format="%.2f", default_value=0.19)
                        dpg.add_input_double(tag="hole_radius_nylon",label="Hole Radius", format="%.7f m", default_value=2.5e-5/2, step=0.1e-5)
                    with dpg.tree_node(label="Mesh Material Properties"):
                        dpg.add_input_double(tag="density_nylon",label="Density", format="%.0f kg/m^3", default_value=1140, step=1)
                        dpg.add_input_double(tag="specific_heat_capacity_nylon",label="Specific Heat Capacity", format="%.0f J/(kg K)", default_value=1670, step=1)

                    with dpg.tree_node(label="Air Material Properties"):
                        dpg.add_input_double(tag="speed_of_sound_air",label="Speed of Sound", format="%.0f m/s", default_value=347, step=1)
                        dpg.add_input_double(tag="density_air",label="Density", format="%.0f kg/m^3", default_value=1.18, step=0.01)
                        dpg.add_input_double(tag="specific_heat_capacity_air",label="Specific Heat Capacity", format="%.0f J/(kg K)", default_value=1006, step=1)
                        dpg.add_input_double(tag="thermal_conductivity_air", label="Thermal Conductivity", format="%.5f W/(m K)", default_value=0.02624, step=0.0001)
                        dpg.add_input_double(tag="dynamic_viscosity_air", label="Dyanmic Viscosity", format= "%.7f Pa s", default_value=1.81e-5, step = 0.0000001)
                        dpg.add_input_double(tag="adiabatic_index_air", label="Adiabatic Index", format="%.1f ", default_value=1.4, step=0.1)
                        
            with dpg.group(width=200):
                dpg.add_input_double(tag="sound_frequency", label="Sound Frequency", max_value=200000.0, format="%.0f Hz", default_value=40000.0, step=1000)
                
                attenuation_text = dpg.add_text("No attenuation calculated", label="Attenuation Np/m",show_label=True)
                absorbtion_text = dpg.add_text("No absorbtion calculated", label="Absorbtion %",show_label=True)

            dpg.add_button(label="Calculate Analytical Attenuation", callback=calculate_attenuation)
            dpg.add_button(label="Calculate Analytical Absorbtion", callback=calculate_absorbtion)

            plot_gradient_physical_model_button = dpg.add_button(label="Plot Physical Acoustics Gradient Model", callback=plot_mesh_gradient,enabled=False)

            save_calib_file = dpg.add_button(label="Save Calibration File")



        with dpg.tab(label="Validation"):

            with dpg.window(tag="view_data_popup",modal=False,show=False,width=1400,height=900,pos=[0,0],label="View Data"):
                with dpg.group(horizontal=True):
                    with dpg.plot(label="Raw Thermal Video", tag="validation_data_heat",height=300,width=800,anti_aliased=False):
                        #dpg.add_plot_legend(location=dpg.mvPlot_Location_SouthEast)
                        dpg.add_plot_axis(dpg.mvXAxis, tag="validation_x_axis",no_tick_labels=False)
                        dpg.add_plot_axis(dpg.mvYAxis, tag="validation_y_axis",no_tick_labels=False)
                    dpg.add_colormap_scale(min_scale=0,max_scale=1,tag="validation_data_legend",height=300,colormap=dpg.mvPlotColormap_Plasma)
                dpg.add_slider_int(label="Frame",max_value=1000, default_value=0, tag="validation_time_slider",show=False,callback=update_validation_thermal_frame, width=800)

                dpg.add_separator()

                with dpg.group(horizontal=True):
                    dpg.add_text("Gradient Model:")
                    dpg.add_radio_button(("Physical Model", "Fit Model"), horizontal=True,label="Gradient Model",tag="gradient_model_selector", callback=update_validation_gradient_model)

                with dpg.group(horizontal=True):
                    with dpg.plot(label="Pressure (Gradient)", tag="validation_grad_heat",height=300,width=800,anti_aliased=False):
                        #dpg.add_plot_legend(location=dpg.mvPlot_Location_SouthEast)
                        dpg.add_plot_axis(dpg.mvXAxis, tag="validation_grad_x_axis",no_tick_labels=False)
                        dpg.add_plot_axis(dpg.mvYAxis, tag="validation_grad_y_axis",no_tick_labels=False)
                    dpg.add_colormap_scale(min_scale=0,max_scale=1,tag="validation_grad_legend",height=300,colormap=dpg.mvPlotColormap_Plasma)
                    with dpg.plot(label="Pressure (Microphone)", tag="validation_mic_heat",height=300,width=800,anti_aliased=False):
                        #dpg.add_plot_legend(location=dpg.mvPlot_Location_SouthEast)
                        dpg.add_plot_axis(dpg.mvXAxis, tag="validation_mic_x_axis",no_tick_labels=False)
                        dpg.add_plot_axis(dpg.mvYAxis, tag="validation_mic_y_axis",no_tick_labels=False)
                    dpg.add_colormap_scale(min_scale=0,max_scale=1,tag="validation_mic_legend",height=300,colormap=dpg.mvPlotColormap_Plasma)
                    with dpg.plot(label="Error", tag="validation_error_heat",height=300,width=800,anti_aliased=False):
                        #dpg.add_plot_legend(location=dpg.mvPlot_Location_SouthEast)
                        dpg.add_plot_axis(dpg.mvXAxis, tag="validation_error_x_axis",no_tick_labels=False)
                        dpg.add_plot_axis(dpg.mvYAxis, tag="validation_error_y_axis",no_tick_labels=False)
                    dpg.add_colormap_scale(min_scale=0,max_scale=1,tag="validation_error_legend",height=300,colormap=dpg.mvPlotColormap_RdBu)#mvPlotColormap_RdBu
                
                with dpg.group(horizontal=True):
                    dpg.add_text("Saturation Model:")
                    dpg.add_radio_button(("Physical Model", "Fit Model"), horizontal=True,label="Steady State Model",tag="saturation_model_selector",callback=update_validation_saturation_model)

                with dpg.group(horizontal=True):
                        with dpg.plot(label="Pressure (Saturation)", tag="validation_sat_heat",height=300,width=800,anti_aliased=False):
                            #dpg.add_plot_legend(location=dpg.mvPlot_Location_SouthEast)
                            dpg.add_plot_axis(dpg.mvXAxis, tag="validation_sat_x_axis",no_tick_labels=False)
                            dpg.add_plot_axis(dpg.mvYAxis, tag="validation_sat_y_axis",no_tick_labels=False)
                        dpg.add_colormap_scale(min_scale=0,max_scale=1,tag="validation_sat_legend",height=300,colormap=dpg.mvPlotColormap_Plasma)
                        with dpg.plot(label="Pressure (Microphone)", tag="validation_mic_heat2",height=300,width=800,anti_aliased=False):
                            #dpg.add_plot_legend(location=dpg.mvPlot_Location_SouthEast)
                            dpg.add_plot_axis(dpg.mvXAxis, tag="validation_mic_x_axis2",no_tick_labels=False)
                            dpg.add_plot_axis(dpg.mvYAxis, tag="validation_mic_y_axis2",no_tick_labels=False)
                        dpg.add_colormap_scale(min_scale=0,max_scale=1,tag="validation_mic_legend2",height=300,colormap=dpg.mvPlotColormap_Plasma)
                        with dpg.plot(label="Error", tag="validation_error_heat2",height=300,width=800,anti_aliased=False):
                            #dpg.add_plot_legend(location=dpg.mvPlot_Location_SouthEast)
                            dpg.add_plot_axis(dpg.mvXAxis, tag="validation_error_x_axis2",no_tick_labels=False)
                            dpg.add_plot_axis(dpg.mvYAxis, tag="validation_error_y_axis2",no_tick_labels=False)
                        dpg.add_colormap_scale(min_scale=0,max_scale=1,tag="validation_error_legend2",height=300,colormap=dpg.mvPlotColormap_RdBu)#mvPlotColormap_RdBu


            with dpg.group(indent=10):
                with dpg.tab_bar(tag="thermal_data_tab_bar2") as tab_bar:

                    with dpg.tab(label="tab 1") as tab2:
                        with dpg.group(horizontal=True,width=600):
                            dpg.add_listbox(tag=str(tab2)+"_list_box_therm2")
                            dpg.add_listbox(tag=str(tab2)+"_list_box_mic2")

                        with dpg.group(horizontal=True):
                            dpg.add_button(label="View Selected Data",enabled=False, tag=str(tab2)+"view_converted_data_button_validation",callback=view_thermal_measurement_validataion)



dpg.set_primary_window("Primary Window", True)

with dpg.theme() as disabled_theme:

    with dpg.theme_component(dpg.mvButton, enabled_state=False):
        dpg.add_theme_color(dpg.mvThemeCol_Text, [80, 80, 80])
        dpg.add_theme_color(dpg.mvThemeCol_Button, [39, 39, 39])
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, [39,39,39])

dpg.bind_theme(disabled_theme)


dpg.show_viewport()

dpg.set_viewport_vsync(True)

#dpg.show_metrics()

demo.show_demo()

#def run_task(loop):
#    loop.run_forever()

#thread = threading.Thread(target=run_task, args=(loop,), daemon=True)
#thread.start()




while(dpg.is_dearpygui_running()):
    dpg.render_dearpygui_frame()  

    if recording:
            frame, frame_status = camera.capture_frame()

            data = frame

            if dpg.does_item_exist("camera_data_plot"):
                dpg.set_value("camera_data_plot", [data]) 
                dpg.configure_item("camera_data_plot",cols=data.shape[1],rows=data.shape[0],scale_min=np.min(data),scale_max=np.max(data))
                dpg.configure_item("camera_data_legend",min_scale=np.min(data),max_scale=np.max(data))
            else:
                dpg.add_heat_series(data,cols=data.shape[1],rows=data.shape[0], parent="camera_y_axis",scale_min=np.min(data),scale_max=np.max(data),format="",tag="camera_data_plot")
                dpg.bind_colormap("camera_data_heat", dpg.mvPlotColormap_Plasma)
                dpg.configure_item("camera_data_legend",min_scale=np.min(data),max_scale=np.max(data))


disconnect_thermal_camera(None,None,None)

dpg.destroy_context()