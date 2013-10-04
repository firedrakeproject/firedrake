from diamond.plugins import register_plugin, cb_decorator

def plugin_applies(xpath):
  if xpath == "/my_root_element":
    return True
  else:
    return False

@cb_decorator
def handle_click(xml, xpath):
  print "xpath == ", xpath
  print "xml[0:80] == ", xml[0:80]
  print "Hello, world!"


register_plugin(plugin_applies, "Test", handle_click)
