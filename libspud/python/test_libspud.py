import libspud

libspud.load_options('test.flml')

#libspud.print_options()

assert libspud.get_option('/timestepping/timestep') == 0.025

assert libspud.get_number_of_children('/geometry') == 5
assert libspud.get_child_name('geometry', 0) == "dimension"

assert libspud.option_count('/problem_type') == 1
assert libspud.have_option('/problem_type')

assert libspud.get_option_type('/geometry/dimension') is int
assert libspud.get_option_type('/problem_type') is str

assert libspud.get_option_rank('/geometry/dimension') == 0
assert libspud.get_option_rank('/physical_parameters/gravity/vector_field::GravityDirection/prescribed/value/constant') == 1

assert libspud.get_option_shape('/geometry/dimension') == (-1, -1)
assert libspud.get_option_shape('/problem_type')[0] > 1
assert libspud.get_option_shape('/problem_type')[1] == -1

assert libspud.get_option('/problem_type') == "multimaterial"
assert libspud.get_option('/geometry/dimension') == 2
libspud.set_option('/geometry/dimension', 3)


assert libspud.get_option('/geometry/dimension') == 3

list_path = '/material_phase::Material1/scalar_field::MaterialVolumeFraction/prognostic/boundary_conditions::LetNoOneLeave/surface_ids'
assert libspud.get_option_shape(list_path) == (4, -1)
assert libspud.get_option_rank(list_path) == 1
assert libspud.get_option(list_path) == [7, 8, 9, 10]

libspud.set_option(list_path, [11, 12, 13, 14, 15])
assert libspud.get_option_shape(list_path) == (5, -1)
assert libspud.get_option_rank(list_path) == 1
assert libspud.get_option(list_path)==[11, 12, 13, 14, 15]

tensor_path = '/material_phase::Material1/tensor_field::DummyTensor/prescribed/value::WholeMesh/anisotropic_asymmetric/constant'
assert libspud.get_option_shape(tensor_path) == (2, 2)
assert libspud.get_option_rank(tensor_path) == 2

assert libspud.get_option(tensor_path)==[[1.0,2.0],[3.0,4.0]]

libspud.set_option(tensor_path, [[5.0,6.0,2.0],[7.0,8.0,1.0]])
assert libspud.get_option_shape(tensor_path) == (2,3)
assert libspud.get_option_rank(tensor_path) == 2

assert(libspud.get_option(tensor_path)==[[5.0, 6.0, 2.0],[7.0, 8.0, 1.0]])

try:
  libspud.add_option('/foo')
  assert False
except libspud.SpudNewKeyWarning, e:
  pass

assert libspud.option_count('/foo') == 1

libspud.set_option('/problem_type', 'helloworld')
assert libspud.get_option('/problem_type') == "helloworld"

try:
  libspud.set_option_attribute('/foo/bar', 'foobar')
  assert False
except libspud.SpudNewKeyWarning, e:
  pass

assert libspud.get_option('/foo/bar') == "foobar"

libspud.delete_option('/foo')
assert libspud.option_count('/foo') == 0

try:
  libspud.get_option('/foo')
  assert False
except libspud.SpudKeyError, e:
  pass

try:
  libspud.get_option('/geometry')
  assert False
except libspud.SpudTypeError, e:
  pass

libspud.write_options('test_out.flml')

libspud.set_option('/test',4)

assert libspud.get_option('/test') == 4

libspud.set_option('/test',[[4.0,2.0,3.0],[2.0,5.0,6.6]])

assert libspud.get_option('/test') == [[4.0,2.0,3.0],[2.0,5.0,6.6]]

libspud.set_option('/test',"Hallo")

assert libspud.get_option('/test') == "Hallo"

libspud.set_option('/test',[1,2,3])

assert libspud.get_option('/test') == [1,2,3]

libspud.set_option('/test',[2.3,3.3])

assert libspud.get_option('/test') == [2.3,3.3]

try:
  libspud.set_option('/test')
  assert False
except libspud.SpudError, e:
  pass


print "All tests passed!"
