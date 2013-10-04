/*  Copyright (C) 2007 Imperial College London and others.

    Please see the AUTHORS file in the main source directory for a full list
    of copyright holders.

    Applied Modelling and Computation Group
    Department of Earth Science and Engineering
    Imperial College London

    David.Ham@Imperial.ac.uk

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation,
    version 2.1 of the License.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307
    USA
*/

#include "spud"

using namespace std;

namespace Spud{

  // OptionManager CLASS METHODS

  // PUBLIC METHODS

  void OptionManager::clear_options() {
    manager.reset();

    return;
  }

  void* OptionManager::get_manager() {
    return (void*) manager.options;
  }

  void OptionManager::set_manager(void* m) {
    delete manager.options;
    manager.options = (Spud::OptionManager::Option*) m;
    return;
  }

  OptionError OptionManager::load_options(const string& filename){
    return manager.options->load_options(filename);
  }

  OptionError OptionManager::write_options(const string& filename){
    return manager.options->write_options(filename);
  }

  OptionError OptionManager::get_child_name(const string& key, const unsigned& index, string& child_name){
    deque<string> kids;
    manager.options->list_children(key, kids);
    if(kids.size() < index){
      return SPUD_KEY_ERROR;
    }

    child_name = kids[index];

    return SPUD_NO_ERROR;
  }

  OptionError OptionManager::get_number_of_children(const string& key, int& child_count){
    deque<string> kids;

    manager.options->list_children(key, kids);

    child_count = kids.size();

    return check_key(key);
  }

  int OptionManager::option_count(const string& key){
    return manager.options->option_count(key);
  }

  logical_t OptionManager::have_option(const string& key){
    return manager.options->have_option(key);
  }

  OptionError OptionManager::get_option_type(const string& key, OptionType& type){
    Option* child = manager.options->get_child(key);
    if(child == NULL){
      return SPUD_KEY_ERROR;
    }

    type = child->get_option_type();

    return SPUD_NO_ERROR;
  }

  OptionError OptionManager::get_option_rank(const string& key, int& rank){
    Option* child = manager.options->get_child(key);
    if(child == NULL){
      return SPUD_KEY_ERROR;
    }

    rank = child->get_option_rank();

    return SPUD_NO_ERROR;
  }

  OptionError OptionManager::get_option_shape(const string& key, vector<int>& shape){
    Option* child = manager.options->get_child(key);
    if(child == NULL){
      return SPUD_KEY_ERROR;
    }

    shape = child->get_option_shape();

    return SPUD_NO_ERROR;
  }

  OptionError OptionManager::get_option(const string& key, double& val){
    OptionError check_err = check_option(key, SPUD_DOUBLE, 0);
    if(check_err != SPUD_NO_ERROR){
      return check_err;
    }

    vector<double> val_handle;
    OptionError get_err = manager.options->get_option(key, val_handle);
    if(get_err != SPUD_NO_ERROR){
      return get_err;
    }else if(val_handle.size() != 1){
      return SPUD_RANK_ERROR;
    }

    val = val_handle[0];

    return SPUD_NO_ERROR;
  }

  OptionError OptionManager::get_option(const string& key, double& val, const double& default_val){
    if(!have_option(key)){
      val = default_val;
      return SPUD_NO_ERROR;
    }

    return get_option(key, val);
  }

  OptionError OptionManager::get_option(const string& key, vector<double>& val){
    OptionError check_err = check_option(key, SPUD_DOUBLE, 1);
    if(check_err != SPUD_NO_ERROR){
      return check_err;
    }

    vector<double> val_handle;
    OptionError get_err = manager.options->get_option(key, val_handle);
    if(get_err != SPUD_NO_ERROR){
      return get_err;
    }

    val = val_handle;

    return SPUD_NO_ERROR;
  }

  OptionError OptionManager::get_option(const string& key, vector<double>& val, const vector<double>& default_val){
    if(!have_option(key)){
      val = default_val;
      return SPUD_NO_ERROR;
    }

    return get_option(key, val);
  }

  OptionError OptionManager::get_option(const string& key, vector< vector<double> >& val){
    OptionError check_err = check_option(key, SPUD_DOUBLE, 2);
    if(check_err != SPUD_NO_ERROR){
      return check_err;
    }

    vector<int> shape;
    OptionError shape_err = get_option_shape(key, shape);
    if(shape_err != SPUD_NO_ERROR){
      return shape_err;
    }

    vector<double> val_handle;
    OptionError get_err = manager.options->get_option(key, val_handle);
    if(get_err != SPUD_NO_ERROR){
      return get_err;
    }

    val.clear();
    for(int i = 0;i < shape[0];i++){
      val.push_back(vector<double>(shape[1]));
      for(int j = 0;j < shape[1];j++){
        val[i][j] = val_handle[(i * shape[1]) + j];
      }
    }

    return SPUD_NO_ERROR;
  }

  OptionError OptionManager::get_option(const string& key, vector< vector<double> >& val, const vector< vector<double> >& default_val){
    if(!have_option(key)){
      val = default_val;
      return SPUD_NO_ERROR;
    }

    return get_option(key, val);
  }

  OptionError OptionManager::get_option(const string& key, int& val){
    OptionError check_err = check_option(key, SPUD_INT, 0);
    if(check_err != SPUD_NO_ERROR){
      return check_err;
    }

    vector<int> val_handle;
    OptionError get_err = manager.options->get_option(key, val_handle);
    if(get_err != SPUD_NO_ERROR){
      return get_err;
    }else if(val_handle.size() != 1){
      return SPUD_RANK_ERROR;
    }

    val = val_handle[0];

    return SPUD_NO_ERROR;
  }

  OptionError OptionManager::get_option(const string& key, int& val, const int& default_val){
    if(!have_option(key)){
      val = default_val;
      return SPUD_NO_ERROR;
    }

    return get_option(key, val);
  }

  OptionError OptionManager::get_option(const string& key, vector<int>& val){
    OptionError check_err = check_option(key, SPUD_INT, 1);
    if(check_err != SPUD_NO_ERROR){
      return check_err;
    }

    vector<int> val_handle;
    OptionError get_err = manager.options->get_option(key, val_handle);
    if(get_err != SPUD_NO_ERROR){
      return get_err;
    }

    val = val_handle;

    return SPUD_NO_ERROR;
  }

  OptionError OptionManager::get_option(const string& key, vector<int>& val, const vector<int>& default_val){
    if(!have_option(key)){
      val = default_val;
      return SPUD_NO_ERROR;
    }

    return get_option(key, val);
  }

  OptionError OptionManager::get_option(const string& key, vector< vector<int> >& val){
    OptionError check_err = check_option(key, SPUD_INT, 2);
    if(check_err != SPUD_NO_ERROR){
      return check_err;
    }

    vector<int> shape;
    OptionError shape_err = get_option_shape(key, shape);
    if(shape_err != SPUD_NO_ERROR){
      return shape_err;
    }

    vector<int> val_handle;
    OptionError get_err = manager.options->get_option(key, val_handle);
    if(get_err != SPUD_NO_ERROR){
      return get_err;
    }

    val.clear();
    for(int i = 0;i < shape[0];i++){
      val.push_back(vector<int>(shape[1]));
      for(int j = 0;j < shape[1];j++){
        val[i][j] = val_handle[(i * shape[1]) + j];
      }
    }

    return SPUD_NO_ERROR;
  }

  OptionError OptionManager::get_option(const string& key, vector< vector<int> >& val, const vector< vector<int> >& default_val){
    if(!have_option(key)){
      val = default_val;
      return SPUD_NO_ERROR;
    }

    return get_option(key, val);
  }

  OptionError OptionManager::get_option(const string& key, string& val){
    OptionError check_err = check_option(key, SPUD_STRING, 1);
    if(check_err != SPUD_NO_ERROR){
      return check_err;
    }

    string val_handle;
    OptionError get_err = manager.options->get_option(key, val_handle);
    if(get_err != SPUD_NO_ERROR){
      return get_err;
    }

    val = val_handle;

    return SPUD_NO_ERROR;
  }

  OptionError OptionManager::get_option(const string& key, string& val, const string& default_val){
    if(!have_option(key)){
      val = default_val;
      return SPUD_NO_ERROR;
    }

    return get_option(key, val);
  }

  OptionError OptionManager::add_option(const string& key){
    logical_t new_key = !have_option(key);

    OptionError add_err = manager.options->add_option(key);
    if(add_err != SPUD_NO_ERROR){
      return add_err;
    }else if(new_key){
      return SPUD_NEW_KEY_WARNING;
    }

    return SPUD_NO_ERROR;
  }

  OptionError OptionManager::set_option(const string& key, const double& val){
    logical_t new_key = !have_option(key);

    vector<double> val_handle;
    val_handle.push_back(val);
    vector<int> shape(2);
    shape[0] = -1;  shape[1] = -1;
    OptionError set_err = manager.options->set_option(key + "/__value", val_handle, 0, shape);
    if(set_err != SPUD_NO_ERROR){
      return set_err;
    }else if(new_key){
      return SPUD_NEW_KEY_WARNING;
    }

    return SPUD_NO_ERROR;
  }

  OptionError OptionManager::set_option(const string& key, const vector<double>& val){
    logical_t new_key = !have_option(key);

    vector<double> val_handle = val;
    vector<int> shape(2);
    shape[0] = val.size();  shape[1] = -1;
    OptionError set_err = manager.options->set_option(key + "/__value", val_handle, 1, shape);
    if(set_err != SPUD_NO_ERROR){
      return set_err;
    }else if(new_key){
      return SPUD_NEW_KEY_WARNING;
    }

    return SPUD_NO_ERROR;
  }

  OptionError OptionManager::set_option(const string& key, const vector< vector<double> >& val){
    logical_t new_key = !have_option(key);

    vector<double> val_handle;
    for(size_t i = 0;i < val.size();i++){
      if(i > 0 and val[i].size() != val[0].size()){
        return SPUD_SHAPE_ERROR;
      }
      for(size_t j = 0;j < val[i].size();j++){
        val_handle.push_back(val[i][j]);
      }
    }
    vector<int> shape(2);
    shape[0] = val.size();
    if(val.size() == 0){
      shape[1] = 0;
    }else{
      shape[1] = val[0].size();
    }
    OptionError set_err = manager.options->set_option(key + "/__value", val_handle, 2, shape);
    if(set_err != SPUD_NO_ERROR){
      return set_err;
    }else if(new_key){
      return SPUD_NEW_KEY_WARNING;
    }

    return SPUD_NO_ERROR;
  }

  OptionError OptionManager::set_option(const string& key, const int& val){
    logical_t new_key = !have_option(key);

    vector<int> val_handle;
    val_handle.push_back(val);
    vector<int> shape(2);
    shape[0] = -1;  shape[1] = -1;
    OptionError set_err = manager.options->set_option(key + "/__value", val_handle, 0, shape);
    if(set_err != SPUD_NO_ERROR){
      return set_err;
    }else if(new_key){
      return SPUD_NEW_KEY_WARNING;
    }

    return SPUD_NO_ERROR;
  }

  OptionError OptionManager::set_option(const string& key, const vector<int>& val){
    logical_t new_key = !have_option(key);

    vector<int> val_handle = val;
    vector<int> shape(2);
    shape[0] = val.size();  shape[1] = -1;
    OptionError set_err = manager.options->set_option(key + "/__value", val_handle, 1, shape);
    if(set_err != SPUD_NO_ERROR){
      return set_err;
    }else if(new_key){
      return SPUD_NEW_KEY_WARNING;
    }

    return SPUD_NO_ERROR;
  }

  OptionError OptionManager::set_option(const string& key, const vector< vector<int> >& val){
    logical_t new_key = !have_option(key);

    vector<int> val_handle;
    for(size_t i = 0;i < val.size();i++){
      if(i > 0 and val[i].size() != val[0].size()){
        return SPUD_SHAPE_ERROR;
      }
      for(size_t j = 0;j < val[i].size();j++){
        val_handle.push_back(val[i][j]);
      }
    }
    vector<int> shape(2);
    shape[0] = val.size();
    if(val.size() == 0){
      shape[1] = 0;
    }else{
      shape[1] = val[0].size();
    }
    OptionError set_err = manager.options->set_option(key + "/__value", val_handle, 2, shape);
    if(set_err != SPUD_NO_ERROR){
      return set_err;
    }else if(new_key){
      return SPUD_NEW_KEY_WARNING;
    }

    return SPUD_NO_ERROR;
  }

  OptionError OptionManager::set_option(const string& key, const string& val){
    logical_t new_key = !have_option(key);

    OptionError set_err = manager.options->set_option(key + "/__value", val);
    if(set_err != SPUD_NO_ERROR){
      return set_err;
    }else if(new_key){
      return SPUD_NEW_KEY_WARNING;
    }

    return SPUD_NO_ERROR;
  }

  OptionError OptionManager::set_option_attr(const string& key, const string& val){
    logical_t new_key = !have_option(key);

    OptionError set_err = manager.options->set_option(key, val);
    if(set_err != SPUD_NO_ERROR){
      return set_err;
    }else if(new_key){
      return SPUD_NEW_KEY_WARNING;
    }

    return SPUD_NO_ERROR;
  }

  OptionError OptionManager::set_option_attribute(const string& key, const string& val){
    OptionError set_err = set_option_attr(key, val);
    if(set_err != SPUD_NO_ERROR and set_err != SPUD_NEW_KEY_WARNING){
      return set_err;
    }

    Option* child = manager.options->get_child(key);
    if(child == NULL){
      return SPUD_KEY_ERROR;
    }
    logical_t is_attribute = child->set_is_attribute(true);
    if(set_err != SPUD_NO_ERROR){
      return set_err;
    }
    else if(!is_attribute){
      return SPUD_ATTR_SET_FAILED_WARNING;
    }

    return SPUD_NO_ERROR;
  }

  OptionError OptionManager::move_option(const string& key1, const string& key2){
    OptionError move_err = manager.options->move_option(key1, key2);
    if(move_err != SPUD_NO_ERROR){
      return move_err;
    }

    return SPUD_NO_ERROR;
  }

  OptionError OptionManager::copy_option(const string& key1, const string& key2){
    OptionError copy_err = manager.options->copy_option(key1, key2);
    if(copy_err != SPUD_NO_ERROR){
      return copy_err;
    }

    return SPUD_NO_ERROR;
  }

  OptionError OptionManager::delete_option(const string& key){
    OptionError del_err = manager.options->delete_option(key);
    if(del_err != SPUD_NO_ERROR){
      return del_err;
    }

    return SPUD_NO_ERROR;
  }

  void OptionManager::print_options(){
    manager.options->print();

    return;
  }

  // PRIVATE METHODS

  OptionManager::OptionManager(){
    options = new Option();
    deallocated = false;

    return;
  }

  OptionManager::OptionManager(const OptionManager& manager){
    cerr << "SPUD ERROR: OptionManager copy constructor cannot be called" << endl;
    exit(-1);
  }

  OptionManager::~OptionManager(){
    if (!deallocated)
    {
      delete options;
      deallocated = true;
    }

    return;
  }

  OptionManager& OptionManager::operator=(const OptionManager& manager){
    cerr << "SPUD ERROR: OptionManager assignment operator cannot be called" << endl;
    exit(-1);
  }

  OptionError OptionManager::check_key(const string& key){
    if(!have_option(key)){
      return SPUD_KEY_ERROR;
    }

    return SPUD_NO_ERROR;
  }

  OptionError OptionManager::check_rank(const string& key, const int& rank){
    int rank_handle;
    OptionError rank_err = get_option_rank(key, rank_handle);
    if(rank_err != SPUD_NO_ERROR){
      return rank_err;
    }else if(rank_handle != rank){
      return SPUD_RANK_ERROR;
    }

    return SPUD_NO_ERROR;
  }

  OptionError OptionManager::check_type(const string& key, const OptionType& type){
    OptionType type_handle;
    OptionError type_err = get_option_type(key, type_handle);
    if(type_err != SPUD_NO_ERROR){
      return type_err;
    }else if(type_handle != type){
      return SPUD_TYPE_ERROR;
    }

    return SPUD_NO_ERROR;
  }

  OptionError OptionManager::check_option(const string& key, const OptionType& type, const int& rank){
    OptionError check_err;
    check_err = check_key(key);
    if(check_err != SPUD_NO_ERROR){
      return check_err;
    }
    check_err = check_type(key, type);
    if(check_err != SPUD_NO_ERROR){
      return check_err;
    }
    check_err = check_rank(key, rank);
    if(check_err != SPUD_NO_ERROR){
      return check_err;
    }

    return SPUD_NO_ERROR;
  }

  void OptionManager::reset(){
    delete options;
    options = new Option;

    return;
  }

  // End OptionManager CLASS METHODS

  // OptionManager::Option CLASS METHODS

  // PUBLIC METHODS

  OptionManager::Option::Option(){
    verbose_off();
    OptionError set_err = set_rank_and_shape(-1, vector<int>());
    if(set_err != SPUD_NO_ERROR){
      cerr << "SPUD ERROR: Failed to set rank and shape" << endl;
      exit(-1);
    }
    is_attribute = false;

    return;
  }

  OptionManager::Option::Option(const OptionManager::Option& inOption){
    *this = inOption;

    return;
  }

  OptionManager::Option::Option(string name){
    verbose_off();
    node_name = name;
    OptionError set_err = set_rank_and_shape(-1, vector<int>());
    if(set_err != SPUD_NO_ERROR){
      cerr << "SPUD ERROR: Failed to set rank and shape" << endl;
      exit(-1);
    }
    is_attribute = false;

    return;
  }

  OptionManager::Option::~Option(){
    return;
  }

  const OptionManager::Option& OptionManager::Option::operator=(const OptionManager::Option& inOption){
    verbose = inOption.verbose;
    if(verbose)
      cout << "const OptionManager::Option& OptionManager::Option::operator=(const OptionManager::Option& inOption)\n";

    node_name = inOption.node_name;
    children = inOption.children;

    data_double = inOption.data_double;
    data_int = inOption.data_int;
    data_string = inOption.data_string;
    vector<int> shape(2);
    shape[0] = inOption.shape[0];  shape[1] = inOption.shape[1];
    OptionError set_err = set_rank_and_shape(inOption.rank, shape);
    if(set_err != SPUD_NO_ERROR){
      cerr << "SPUD ERROR: Failed to set rank and shape" << endl;
      exit(-1);
    }

    is_attribute = inOption.is_attribute;

    return *this;
  }

  OptionError OptionManager::Option::load_options(const string& filename){
    if(verbose)
      cout << "void OptionManager::Option::load_options(const string& filename = " << filename << ")\n";

    delete_option("/");

    TiXmlDocument doc(filename);
    doc.SetCondenseWhiteSpace(false);
    if(!doc.LoadFile()){
      //cerr << "SPUD WARNING: Failed to load options file " << filename << endl;
      return SPUD_FILE_ERROR;
    }

    TiXmlNode* header = doc.FirstChild();
    while(header != NULL and header->Type() != TiXmlNode::DECLARATION){
      header = header->NextSibling();
    }
    TiXmlNode* fluidity_options = doc.FirstChildElement();
    if(fluidity_options == NULL){
      //cerr << "SPUD WARNING: Failed to find root node when loading options file" << endl;
      return SPUD_FILE_ERROR;
    }

    // Set the name of this element
    node_name = fluidity_options->ValueStr();

    // Decend down through all the fluidity options
    TiXmlNode* option_node;
    for(option_node = fluidity_options->FirstChildElement();option_node;option_node = option_node->NextSiblingElement()){
      if(option_node->ToElement()){
        parse_node("", option_node);
      }
    }

    return SPUD_NO_ERROR;
  }

  OptionError OptionManager::Option::write_options(const string& filename) const{
    if(verbose)
      cout << "void OptionManager::Option::write_options(const string& filename = " << filename << ") const\n";

    TiXmlDocument doc;

    // XML header
    TiXmlDeclaration* header = new TiXmlDeclaration("1.0", "utf-8", "");
    doc.LinkEndChild(header);

    // Root node
    TiXmlNode* root_node = to_element();
    doc.LinkEndChild(root_node);

    if(!doc.SaveFile(filename)){
      return SPUD_FILE_ERROR;
      //cerr << "SPUD WARNING: Failed to write options file" << endl;
    }

    return SPUD_NO_ERROR;
  }

  string OptionManager::Option::get_name() const{
    if(verbose)
      cout << "void OptionManager::Option::get_name(void) const\n";
    return node_name;
  }

  logical_t OptionManager::Option::get_is_attribute() const{
    if(verbose)
      cout << "logical_t OptionManager::Option::get_is_attribute(void) const\n";
    return is_attribute;
  }

  void OptionManager::Option::list_children(const string& name, deque<string>& kids) const{
    if(verbose)
      cout << "void OptionManager::Option::list_children(const string& name = " << name << ", deque<string>& kids) const\n";

    kids.clear();

    const Option* descendant = get_child(name);
    if(descendant != NULL){
      for(deque< pair<string, Option> >::const_iterator it = descendant->children.begin();it != descendant->children.end();it++){
        kids.push_back(it->first);
      }
    }

    return;
  }

  size_t OptionManager::Option::count(const string& key) const{
    size_t cnt=0;
    for(deque< pair<string, Option> >::const_iterator it=children.begin();it!=children.end();++it){
      if(it->first == key)
        cnt++;
    }
    return cnt;
  }

  deque< pair<string, OptionManager::Option> >::iterator OptionManager::Option::find(const string& key){
    for(deque< pair<string, Option> >::iterator it=children.begin();it!=children.end();++it){
      if(it->first == key)
        return it;
    }
    return children.end();
  }

  deque< pair<string, OptionManager::Option> >::const_iterator OptionManager::Option::find(const string& key) const{
    for(deque< pair<string, Option> >::const_iterator it=children.begin();it!=children.end();++it){
      if(it->first == key)
        return it;
    }
    return children.end();
  }

  deque< pair<string, OptionManager::Option> >::iterator OptionManager::Option::find_next(deque< pair<string, OptionManager::Option> >::iterator current, const string& key){
    deque< pair<string, OptionManager::Option> >::iterator it = current;
    it++;
    for(;it!=children.end();++it){
      if(it->first == key)
        return it;
    }
    return children.end();
  }

  deque< pair<string, OptionManager::Option> >::const_iterator OptionManager::Option::find_next(deque< pair<string, OptionManager::Option> >::const_iterator current, const string& key) const{
    deque< pair<string, Option> >::const_iterator it = current;
    it++;
    for(;it!=children.end();++it){
      if(it->first == key)
        return it;
    }
    return children.end();
  }

  const OptionManager::Option* OptionManager::Option::get_child(const string& key) const{
    if(verbose)
      cout << "const OptionManager::Option* OptionManager::Option::get_child(const string& key = " << key << ") const\n";

    if(key == "/" or key.empty())
      return this;

    string name, branch;
    int index;
    OptionError key_err = split_name(key, name, index, branch);
    if(key_err != SPUD_NO_ERROR){
      return NULL;
    }

    if(name.empty()){
      return NULL;
    }

    deque< pair<string, Option> >::const_iterator it;
    if(count(name) == 0){
      name += "::";
      int i = 0;
      for(it = children.begin();it != children.end();it++){
        if(it->first.compare(0, name.size(), name) == 0){
          if(index < 0 or i == index){
            break;
          }else{
            i++;
          }
        }
      }
    }else{
      if(index < 0){
        it = find(name);
      }else{
        it = find(name);
        for(size_t i = 0;it!=children.end();i++){
          if((int)i == index){
            break;
          }
          it = find_next(it, name);
        }
      }
    }

    if(it == children.end()){
      return NULL;
    }else if(branch.empty()){
      return &(it->second);
    }else{
      return it->second.get_child(branch);
    }
  }

  OptionManager::Option* OptionManager::Option::get_child(const string& key){
    if(verbose)
      cout << "OptionManager::Option* OptionManager::Option::get_child(const string& key = " << key <<")\n";

    if(!have_option(key)){
      return NULL;
    }else{
      return create_child(key);
    }
  }

  int OptionManager::Option::option_count(const string& key) const{
    if(verbose)
      cout << "int OptionManager::Option::option_count(const string& key = " << key << ") const\n";

    string name, branch;
    int index;
    OptionError key_err = split_name(key, name, index, branch);
    if(key_err != SPUD_NO_ERROR){
      return 0;
    }

    if(name.empty()){
      return 0;
    }

    logical_t match_whole = true;
    if(!count(name)){
      // Apparently there is no such child but lets check for "name::*"
      name += "::";
      match_whole = false;
    }

    int count = 0, i = 0;
    for(deque< pair<string, Option> >::const_iterator it = children.begin();it != children.end();it++){
      if(it->first.compare(0, name.size(), name) == 0 and (not match_whole or it->first.size() == name.size())){
        if(index < 0){
          if(branch.empty()){
            count++;
          }else{
            count += it->second.option_count(branch);
          }
        }else{
          if(i == index){
            if(branch.empty()){
              count++;
            }else{
              count += it->second.option_count(branch);
            }
            break;
          }
          i++;
        }
      }
    }

    return count;
  }

  logical_t OptionManager::Option::have_option(const string& key) const{
    if(verbose)
      cout << "logical_t OptionManager::Option::have_option(const string& key = " << key << ") const\n";

    if(key == "/"){
      return true;
    }else{
      return (get_child(key) != NULL);
    }
  }

  OptionType OptionManager::Option::get_option_type() const{
    if(verbose)
      cout << "OptionType OptionManager::Option::get_option_type(void) const\n";

    if(have_option("__value")){
      return find("__value")->second.get_option_type();
    }

    if(!data_double.empty()){
      return SPUD_DOUBLE;
    }else if(!data_int.empty()){
      return SPUD_INT;
    }else if(!data_string.empty()){
      return SPUD_STRING;
    }else{
      return SPUD_NONE;
    }
  }

  size_t OptionManager::Option::get_option_rank() const{
    if(verbose)
      cout << "size_t OptionManager::Option::get_option_rank(void) const\n";

    if(have_option("__value")){
      return find("__value")->second.get_option_rank();
    }else{
      return rank;
    }
  }

  vector<int> OptionManager::Option::get_option_shape() const{
    if(verbose)
      cout << "vector<int> OptionManager::Option::get_option_shape(void) const\n";

    if(have_option("__value")){
      return find("__value")->second.get_option_shape();
    }else{
      vector<int> shape(2);
      shape[0] = this->shape[0];
      shape[1] = this->shape[1];
      return shape;
    }
  }

  OptionError OptionManager::Option::get_option(vector<double>& val) const{
    if(verbose)
      cout << "OptionError OptionManager::Option::get_option(vector<double>& val) const\n";

    if(have_option("__value")){
      return get_option("__value", val);
    }else if(get_option_type() != SPUD_DOUBLE){
      return SPUD_TYPE_ERROR;
    }else{
      val = data_double;
      return SPUD_NO_ERROR;
    }
  }

  OptionError OptionManager::Option::get_option(vector<int>& val) const{
    if(verbose)
      cout << "OptionError OptionManager::Option::get_option(vector<int>& val) const\n";

    if(have_option("__value")){
      return get_option("__value", val);
    }else if(get_option_type() != SPUD_INT){
      return SPUD_TYPE_ERROR;
    }else{
      val = data_int;
      return SPUD_NO_ERROR;
    }
  }

  OptionError OptionManager::Option::get_option(string& val) const{
    if(verbose)
      cout << "OptionError OptionManager::Option::get_option(string& val = " << val << ") const\n";

    if(have_option("__value")){
      return get_option("__value", val);
    }else if(get_option_type() != SPUD_STRING){
      return SPUD_TYPE_ERROR;
    }else{
      val = data_string;
      return SPUD_NO_ERROR;
    }
  }

  OptionError OptionManager::Option::get_option(const string& key, vector<double>& val) const{
    if(verbose)
      cout << "OptionError OptionManager::Option::get_option(const string& key = " << key << ", vector<double>& val)\n";

    const Option* child = get_child(key);
    if(child == NULL){
      return SPUD_KEY_ERROR;
    }else{
      return child->get_option(val);
    }
  }

  OptionError OptionManager::Option::get_option(const string& key, vector<int>& val) const{
    if(verbose)
      cout << "OptionError OptionManager::Option::get_option(const string& key = " << key << ", vector<int>& val)\n";

    const Option* child = get_child(key);
    if(child == NULL){
      return SPUD_KEY_ERROR;
    }else{
      return child->get_option(val);
    }
  }

  OptionError OptionManager::Option::get_option(const string& key, string& val) const{
    if(verbose)
      cout << "OptionError OptionManager::Option::get_option(const string& key = " << key << ", string& val = " << val << ")\n";

    const Option* child = get_child(key);
    if(child == NULL){
      return SPUD_KEY_ERROR;
    }else{
      return child->get_option(val);
    }
  }

  OptionError OptionManager::Option::add_option(const string& key){
    if(verbose)
      cout << "OptionError OptionManager::Option::add_option(const string& key = " << key << ")\n";

    return create_child(key) == NULL ? SPUD_KEY_ERROR : SPUD_NO_ERROR;
  }

  OptionError OptionManager::Option::set_option(const vector<double>& val, const int& rank, const vector<int>& shape){
    if(verbose)
      cout << "OptionError OptionManager::Option::set_option(const vector<double>& val, const int& rank = " << rank << ", const vector<int>& shape)\n";

    if(have_option("__value")){
      return set_option("__value", val, rank, shape);
    }else{
      data_double = val;
      OptionError set_err = set_option_type(SPUD_DOUBLE);
      if(set_err != SPUD_NO_ERROR){
        return set_err;
      }
      set_err = set_rank_and_shape(rank, shape);
      if(set_err != SPUD_NO_ERROR){
        return set_err;
      }
      return SPUD_NO_ERROR;
    }
  }

  OptionError OptionManager::Option::set_option(const vector<int>& val, const int& rank, const vector<int>& shape){
    if(verbose)
      cout << "OptionError OptionManager::Option::set_option(const vector<int>& val, const int& rank = " << rank << ", const vector<int>& shape)\n";

    if(have_option("__value")){
      return set_option("__value", val, rank, shape);
    }else{
      data_int = val;
      OptionError set_err = set_option_type(SPUD_INT);
      if(set_err != SPUD_NO_ERROR){
        return set_err;
      }
      set_err = set_rank_and_shape(rank, shape);
      if(set_err != SPUD_NO_ERROR){
        return set_err;
      }
      return SPUD_NO_ERROR;
    }
  }

  OptionError OptionManager::Option::set_option(const string& val){
    if(verbose)
      cout << "OptionError OptionManager::Option::set_option(const string& val = " << val << ")\n";

    if(have_option("__value")){
      return set_option("__value", val);
    }else{
      data_string = val;
      vector<int> shape(2);
      shape[0] = val.size();  shape[1] = -1;
      OptionError set_err = set_option_type(SPUD_STRING);
      if(set_err != SPUD_NO_ERROR){
        return set_err;
      }
      set_err = set_rank_and_shape(1, shape);
      if(set_err != SPUD_NO_ERROR){
        return set_err;
      }
      return SPUD_NO_ERROR;
    };
  }

  OptionError OptionManager::Option::set_option(const string& key, const vector<double>& val, const int& rank, const vector<int>& shape){
    if(verbose)
      cout << "OptionError OptionManager::Option::set_option(const string& key = " << key << ", const vector<double>& val, const int& rank = " << rank << ", const vector<int>& shape)\n";

    Option* opt = create_child(key);
    if(opt == NULL){
      return SPUD_KEY_ERROR;
    }else{
      return opt->set_option(val, rank, shape);
    }
  }

  OptionError OptionManager::Option::set_option(const string& key, const vector<int>& val, const int& rank, const vector<int>& shape){
    if(verbose)
      cout << "OptionError OptionManager::Option::set_option(const string& key = " << key << ", const vector<int>& val, const int& rank = " << rank << ", const vector<int>& shape)\n";

    Option* opt = create_child(key);
    if(opt == NULL){
      return SPUD_KEY_ERROR;
    }else{
      return opt->set_option(val, rank, shape);
    }
  }

  OptionError OptionManager::Option::set_option(const string& key, const string& val){
    if(verbose)
      cout << "OptionError OptionManager::Option::set_option(const string& key = " << key << ", const string& val = " << val <<")\n";

    Option* opt = create_child(key);
    if(opt == NULL){
      return SPUD_KEY_ERROR;
    }else{
      return opt->set_option(val);
    }
  }

  logical_t OptionManager::Option::set_is_attribute(const logical_t& is_attribute){
    if(verbose)
      cout << "logical_t OptionManager::Option::set_is_attribute(const logical_t& is_attribute = " << is_attribute << ")\n";
    if(children.size() == 0 and get_option_type() == SPUD_STRING){
      this->is_attribute = is_attribute;
    }

    return this->is_attribute;
  }

  OptionError OptionManager::Option::set_attribute(const string& key, const string& val){
    if(verbose)
      cout << "OptionError OptionManager::Option::set_attribute(const string& key = " << key << ", const string& val = " << val << ")\n";

    Option* opt = create_child(key);
    if(opt == NULL){
      return SPUD_KEY_ERROR;
    }else{
      OptionError set_err = opt->set_option(val);
      if(set_err != SPUD_NO_ERROR){
        return set_err;
      }
      opt->set_is_attribute(true);
      return SPUD_NO_ERROR;
    }
  }

  OptionError OptionManager::Option::copy_option(const string& key1, const string& key2){
    if(verbose)
      cout << "OptionError OptionManager::Option::copy_option(const string& key1 = " << key1 << ", const string& key2 = " << key2 << ")\n";

    const Option* option1 = get_child(key1);
    if(option1 == NULL){
      return SPUD_KEY_ERROR;
    }

    const Option* option2 = get_child(key2);
    if(option2 != NULL){
      return SPUD_KEY_ERROR;
    }

    string::size_type lastPos = key2.find_last_not_of("/");
    lastPos = key2.find_last_of("/", lastPos);
    string key2_parent = key2.substr(0, lastPos);
    string key2_name = key2.substr(lastPos + 1);

    Option* option2_parent = create_child(key2_parent);
    if(option2_parent == NULL){
      return SPUD_KEY_ERROR;
    }

    Option new_option1 = Option(*option1);
    new_option1.node_name = key2_name;
    string new_node_name, name_attr;
    new_option1.split_node_name(new_node_name, name_attr);
    if(name_attr.size() == 0){
      option2_parent->children.push_back(pair<string, Option>(new_node_name, new_option1));
    }else{
      new_option1.set_attribute("name", name_attr);
      option2_parent->children.push_back(pair<string, Option>(new_node_name + "::" + name_attr, new_option1));
    }

    return SPUD_NO_ERROR;
  }

  OptionError OptionManager::Option::move_option(const string& key1, const string& key2){
    if(verbose)
      cout << "OptionError OptionManager::Option::move_option(const string& key1 = " << key1 << ", const string& key2 = " << key2 << ")\n";

    const Option* option1 = get_child(key1);
    if(option1 == NULL){
      return SPUD_KEY_ERROR;
    }

    const Option* option2 = get_child(key2);
    if(option2 != NULL){
      return SPUD_KEY_ERROR;
    }

    string::size_type lastPos = key2.find_last_not_of("/");
    lastPos = key2.find_last_of("/", lastPos);
    string key2_parent = key2.substr(0, lastPos);
    string key2_name = key2.substr(lastPos + 1);

    Option* option2_parent = create_child(key2_parent);
    if(option2_parent == NULL){
      return SPUD_KEY_ERROR;
    }

    Option new_option1 = Option(*option1);
    new_option1.node_name = key2_name;
    string new_node_name, name_attr;
    new_option1.split_node_name(new_node_name, name_attr);
    if(name_attr.size() == 0){
      option2_parent->children.push_back(pair<string, Option>(new_node_name, new_option1));
    }else{
      new_option1.set_attribute("name", name_attr);
      option2_parent->children.push_back(pair<string, Option>(new_node_name + "::" + name_attr, new_option1));
    }

    delete_option(key1);

    return SPUD_NO_ERROR;
  }

  OptionError OptionManager::Option::delete_option(const string& key){
    if(verbose)
      cout << "OptionError OptionManager::Option::delete_option(const string& key = " << key << ")\n";

    string branch, name;
    OptionError key_err = split_name(key, name, branch);
    if(key_err != SPUD_NO_ERROR){
      return key_err;
    }

    Option* opt = get_child(name);
    if(opt == NULL){
      return SPUD_KEY_ERROR;
    }else if(branch.empty()){
      deque< pair<string, Option> >::iterator iter = find(name);
      while(iter!=children.end()){
        if(&iter->second == opt){
          children.erase(iter);
          return SPUD_NO_ERROR;
        }
        iter = find_next(iter, name);
      }
      for(iter = children.begin();iter != children.end();iter++){
        if(&iter->second == opt){
          children.erase(iter);
          return SPUD_NO_ERROR;
        }
      }
      return SPUD_KEY_ERROR;
    }else{
      return opt->delete_option(branch);
    }
  }

  void OptionManager::Option::print(const string& prefix) const{
    cout << prefix << node_name;
    string lprefix = prefix + " ";

    if(children.empty()){
      cout << ": ";
      if(!data_double.empty()){
        for(vector<double>::const_iterator i = data_double.begin();i != data_double.end();++i){
          cout << *i << " ";
        }
      }else if(!data_int.empty()){
        for(vector<int>::const_iterator i=data_int.begin();i != data_int.end();++i){
          cout << *i << " ";
        }
      }else if(!data_string.empty()){
        cout << data_string;
      }else{
        cout << "NULL";
      }
      cout << endl;
    }else{
      cout << "/" << endl;

      if(!data_double.empty()){
        cout << lprefix << "<value>: ";
        for(vector<double>::const_iterator i = data_double.begin();i != data_double.end();++i){
          cout << *i<< " ";
        }
        cout << endl;
      }else if(!data_int.empty()){
        cout << lprefix << "<value>: ";
        for(vector<int>::const_iterator i=data_int.begin();i!=data_int.end();++i){
          cout << *i << " ";
        }
        cout << endl;
      }else if(!data_string.empty()){
        cout << lprefix << "<value>: " << data_string;
        cout << endl;
      }
      for(deque< pair<string, Option> >::const_iterator i = children.begin();i!=children.end();++i){
        i->second.print(lprefix + " ");
      }
    }

    return;
  }

  void OptionManager::Option::verbose_on(){
    cout << "void OptionManager::Option::verbose_on(void)\n";

    verbose = true;
  }

  void OptionManager::Option::verbose_off(){
    verbose = false;
  }

  // PRIVATE METHODS

  OptionManager::Option* OptionManager::Option::create_child(const string& key){
    if(verbose)
      cout << "OptionManager::Option* OptionManager::Option::create_child(const string& key = " << key << ")\n";

    if(key == "/" or key.empty())
      return this;

    string branch, name;
    int index;
    OptionError key_err = split_name(key, name, index, branch);
    if(key_err != SPUD_NO_ERROR){
      return NULL;
    }

    if(name.empty()){
      return NULL;
    }

    deque< pair<string, Option> >::iterator child;
    if(count(name) == 0){
      string name2 = name + "::";
      int i = 0;
      for(child = children.begin();child != children.end();child++){
        if(child->first.compare(0, name2.size(), name2) == 0){
          if(index < 0 or i == index){
            break;
          }else{
            i++;
          }
        }
      }
      if(child == children.end()){
        if(name == "__value" and get_option_type() != SPUD_NONE){
          cerr << "SPUD WARNING: Creating __value child for non null element - deleting parent data" << endl;
          set_option_type(SPUD_NONE);
        }
        children.push_back(pair<string, Option>(name, Option(name)));
        string new_node_name, name_attr;
        children.rbegin()->second.split_node_name(new_node_name, name_attr);
        if(name_attr.size() > 0){
          children.rbegin()->second.set_attribute("name", name_attr);
        }
        is_attribute = false;
      }
    }else{
      if(index < 0){
        child = find(name);
      }else{
        child = find(name);
        size_t i;
        for(i = 0;child != children.end();i++){
          if((int)i == index){
            break;
          }
          child = find_next(child, name);
        }
        if(child == children.end() and index == (int)i){
          children.push_back(pair<string, Option>(name, Option(name)));
          child = children.end(); child--;
          is_attribute = false;
        }
      }
    }

    if(child == children.end()){
      return NULL;
    }else if(branch.empty()){
      return &child->second;
    }else{
      return child->second.create_child(branch);
    }
  }

  OptionError OptionManager::Option::set_rank_and_shape(const int& rank, const vector<int>& shape){
    if(verbose)
      cout << "OptionError OptionManager::Option::set_rank_and_shape(const int& rank = " << rank << ", const vector<int>& shape)\n";

    if((int)shape.size() != 2 and rank != -1 and (int)shape.size() != rank){
      return SPUD_SHAPE_ERROR;
    }

    OptionType type = get_option_type();
    logical_t set_attrs = (type == SPUD_DOUBLE or type == SPUD_INT);
    switch(rank){
      case(-1):
        this->rank = -1;
        this->shape[0] = -1;
        this->shape[1] = -1;
        break;
      case(0):
        this->rank = 0;
        this->shape[0] = -1;
        this->shape[1] = -1;
        if(set_attrs){
          ostringstream rank_as_string;
          rank_as_string << this->rank;
          set_attribute("rank", rank_as_string.str());
        }
        break;
      case(1):
        this->rank = 1;
        this->shape[0] = shape[0];
        this->shape[1] = -1;
        if(set_attrs){
          ostringstream rank_as_string;
          rank_as_string << this->rank;
          set_attribute("rank", rank_as_string.str());
          ostringstream shape_as_string;
          shape_as_string << shape[0];
          set_attribute("shape", shape_as_string.str());
        }
        break;
      case(2):
        this->rank = 2;
        this->shape[0] = shape[0];
        this->shape[1] = shape[1];
        if(set_attrs){
          ostringstream rank_as_string;
          rank_as_string << this->rank;
          set_attribute("rank", rank_as_string.str());
          ostringstream shape_as_string;
          shape_as_string << shape[0] << " " << shape[1];
          set_attribute("shape", shape_as_string.str());
        }
        break;
      default:
        return SPUD_RANK_ERROR;
    }

    return SPUD_NO_ERROR;
  }

  OptionError OptionManager::Option::set_option_type(const OptionType& option_type){
    if(verbose)
      cout << "OptionError OptionManager::Option::set_option_type(const OptionType& option_type)\n";

    switch(option_type){
      case(SPUD_DOUBLE):
        data_int.clear();
        data_string = "";
        is_attribute = false;
        break;
      case(SPUD_INT):
        data_double.clear();
        data_string = "";
        is_attribute = false;
        break;
      case(SPUD_NONE):
        data_double.clear();
        data_int.clear();
        data_string = "";
        is_attribute = false;
        break;
      case(SPUD_STRING):
        data_double.clear();
        data_int.clear();
        break;
      default:
        return SPUD_TYPE_ERROR;
    }

    return SPUD_NO_ERROR;
  }

  void OptionManager::Option::parse_node(const string& root, const TiXmlNode* node){
    if(verbose)
      cout << "void OptionManager::Option::parse_node(const string& root = " << root << ", const TiXmlNode* node)\n";

    // Should only deal with TiXmlNode::ELEMENT types
    if(node->Type() != TiXmlNode::ELEMENT){
      cerr << "SPUD WARNING - Non-element: " << root << " encountered" << endl;
      return;
    }

    const TiXmlElement* element = node->ToElement();

    // Establish new base name of this node
    string basename = root + "/" + node->ValueStr();
    if(element->Attribute("name")){
      basename = basename + "::" + element->Attribute("name");
    }

    // Ensure this path has been added
    Option* child = create_child(basename);
    if(child == NULL){
      cerr << "SPUD ERROR: Unexpected failure when creating child element" << endl;
      exit(-1);
    }

    // Store node attributes
    for(const TiXmlAttribute* att = element->FirstAttribute();att;att = att->Next()){
      string att_name(basename + "/" + att->Name());
      set_attribute(att_name, att->ValueStr());
    }

    // Loop through all child elements
    for(const TiXmlNode* cnode = node->FirstChild();cnode;cnode = cnode->NextSibling()){
      switch(cnode->Type()){
        case(TiXmlNode::ELEMENT):
          break;
        case(TiXmlNode::TEXT):
          // Store node data
          set_option(basename, cnode->ValueStr());
          continue;
        default:
          continue;
      }

      // Special case when the child of cnode is empty
      if(!cnode->FirstChild()){
        parse_node(basename, cnode);
        continue;
      }

      const TiXmlElement* celement = cnode->ToElement();

      if(cnode->ValueStr() == string("integer_value")){
        // Tokenise the data stored and convert to ints
        vector<string> tokens;
        tokenize(cnode->FirstChild()->ValueStr(), tokens);

        // Find shape and rank
        int rank;
        vector<int> shape(2);
        istringstream(celement->Attribute("rank")) >> rank;
        if(rank == 0){
          shape[0] = -1;  shape[1] = -1;
        }else if(rank == 1){
          shape[0] = tokens.size();  shape[1] = -1;
        }else if(rank == 2){
          istringstream(celement->Attribute("shape")) >> shape[0] >> shape[1];
        }

        vector<int> val;
        val.resize(tokens.size());
        for(size_t i = 0;i<tokens.size();i++){
          istringstream(tokens[i]) >> val[i];
        }

        set_option(basename + "/__value", val, rank, shape);
        for(const TiXmlAttribute* att = celement->FirstAttribute();att;att = att->Next()){
          string att_name(basename + "/__value/" + att->Name());
          set_attribute(att_name, att->ValueStr());
        }
      }else if(cnode->ValueStr() == string("real_value")){
        // Tokenise the data stored and convert to doubles
        vector<string> tokens;
        tokenize(cnode->FirstChild()->ValueStr(), tokens);

        // Find shape and rank
        int rank;
        vector<int> shape(2);
        istringstream(celement->Attribute("rank")) >> rank;
        if(rank == 0){
          shape[0] = -1;  shape[1] = -1;
        }else if(rank == 1){
          shape[0] = tokens.size();  shape[1] = -1;
        }else if(rank == 2){
          istringstream(celement->Attribute("shape")) >> shape[0] >> shape[1];
        }

        vector<double> val;
        val.resize(tokens.size());
        for(size_t i = 0;i<tokens.size();i++){
          for(size_t j = 0;j < tokens[i].size();j++){
            tokens[i][j] = tolower(tokens[i][j]);
          }
          if(tokens[i] == "nan"){
            val[i] = numeric_limits<double>::quiet_NaN();
          }else if(tokens[i] == "inf"){
            val[i] = numeric_limits<double>::infinity();
          }else if(tokens[i] == "-inf"){
            val[i] = -numeric_limits<double>::infinity();
          }else{
            istringstream(tokens[i]) >> val[i];
          }
        }

        set_option(basename + "/__value", val, rank, shape);
        for(const TiXmlAttribute* att = celement->FirstAttribute();att;att = att->Next()){
          string att_name(basename + "/__value/" + att->Name());
          set_attribute(att_name, att->ValueStr());
        }
      }else if(cnode->ValueStr() == string("string_value")){
        set_option(basename + "/__value", cnode->FirstChild()->ValueStr());
        for(const TiXmlAttribute* att = celement->FirstAttribute();att;att = att->Next()){
          string att_name(basename + "/__value/" + att->Name());
          set_attribute(att_name, att->ValueStr());
        }
      }else{
        parse_node(basename, cnode);
      }
    }
  }

  TiXmlElement* OptionManager::Option::to_element() const{
    if(verbose)
      cout << "TiXmlElement* OptionManager::Option:to_element(void) const\n";

    if(is_attribute){
      cerr << "SPUD WARNING: Converting an attribute to an element" << endl;
    }

    // Create new element
    TiXmlElement* ele = new TiXmlElement(node_name);

    // Set element name and name attribute if composite name
    string node_name, name_attr;
    split_node_name(node_name, name_attr);
    if(name_attr.size() > 0){
      ele->SetValue(node_name);
      ele->SetAttribute("name", name_attr);
    }

    // Set data
    TiXmlText* data_ele = new TiXmlText("");
    data_ele->SetValue(data_as_string());
    ele->LinkEndChild(data_ele);

    for(deque< pair<string, Option> >::const_iterator iter = children.begin();iter != children.end();iter++){
      if(iter->second.is_attribute){
        // Add attribute
        ele->SetAttribute(iter->second.node_name, iter->second.data_as_string());
      }else{
        TiXmlElement* child_ele = iter->second.to_element();
        if(iter->second.node_name == "__value"){
          // Detect data sub-element
          switch(iter->second.get_option_type()){
            case(SPUD_DOUBLE):
              child_ele->SetValue("real_value");
              break;
            case(SPUD_INT):
              child_ele->SetValue("integer_value");
              break;
            case(SPUD_NONE):
              break;
            case(SPUD_STRING):
              child_ele->SetValue("string_value");
              break;
            default:
              cerr << "SPUD ERROR: Invalid option type" << endl;
              exit(-1);
           }
        }
        // Add child
        ele->LinkEndChild(child_ele);
      }
    }
    return ele;
  }

  void OptionManager::Option::tokenize(const string& str, vector<string>& tokens, const string& delimiters) const{
    if(verbose)
      cout << "void OptionManager::Option::tokenize(const string& str = " << str << ", vector<string>& tokens, const string& delimiters = " << delimiters << ")\n";

    tokens.clear();

    // Skip delimiters at beginning.
    string::size_type lastPos = str.find_first_not_of(delimiters, 0);

    // Find first "non-delimiter".
    string::size_type pos     = str.find_first_of(delimiters, lastPos);

    while (string::npos != pos || string::npos != lastPos){
      // Found a token, add it to the vector.
      tokens.push_back(str.substr(lastPos, pos - lastPos));

      // Skip delimiters.  Note the "not_of"
      lastPos = str.find_first_not_of(delimiters, pos);

      // Find next "non-delimiter"
      pos = str.find_first_of(delimiters, lastPos);
    }

    return;
  }

  OptionError OptionManager::Option::split_name(const string& in, string& name, string& branch) const{
    if(verbose)
      cout << "OptionError OptionManager::Option::split_name(const string& in = " << in << ", string& name, string& branch) const\n";

    name = "";
    branch = "";

    string fullname = in.substr(0, min(in.size(), in.find_first_of(" ")));

    // Skip delimiters at beginning.
    string::size_type lastPos = fullname.find_first_not_of("/", 0);
    if(lastPos == string::npos){
      return SPUD_NO_ERROR;
    }

    // Find next delimiter
    string::size_type pos = fullname.find_first_of("/", lastPos);
    if(pos == string::npos){
      name = fullname.substr(lastPos, fullname.size() - lastPos);
    }else{
      name = fullname.substr(lastPos, pos - lastPos);
      branch = fullname.substr(pos, fullname.size()-pos);
    }

    return SPUD_NO_ERROR;
  }

  OptionError OptionManager::Option::split_name(const string& in, string& name, int& index, string& branch) const{
    if(verbose)
      cout << "OptionError OptionManager::Option::split_name(const string& in = " << in << ", string& name, int& index, string& branch) const\n";

    index = -1;
    OptionError key_err = split_name(in, name, branch);
    if(key_err != SPUD_NO_ERROR){
      return key_err;
    }

    // Extract the index from the name if necessary
    string::size_type pos = name.find_first_of("[", 0);
    string::size_type lastPos = name.find_first_of("]", 0);
    if(lastPos < name.size() - 1){
      return SPUD_KEY_ERROR;
    }
    if((lastPos-pos) > 0){
      istringstream(name.substr(pos + 1, lastPos - 1))>>index;
      name = name.substr(0, pos);
    }

    return SPUD_NO_ERROR;
  }

  void OptionManager::Option::split_node_name(string& node_name, string& name_attr) const{
    if(verbose)
      cout << "void OptionManager::Option::split_node_name(string& node_name, string& name_attr) const\n";

    string::size_type firstPos = this->node_name.rfind("::");
    if(firstPos == string::npos or firstPos == this->node_name.size() - 2){
      node_name = this->node_name;
      name_attr = "";
    }else{
      node_name = this->node_name.substr(0, firstPos);
      name_attr = this->node_name.substr(firstPos + 2);
    }

    return;
  }

  string OptionManager::Option::data_as_string() const{
    if(verbose)
      cout << "string OptionManager::Option::data_as_string(void) const\n";

    ostringstream data_as_string;
    data_as_string.precision(numeric_limits< double >::digits10);
    switch(get_option_type()){
      case(SPUD_DOUBLE):
        for(unsigned int i = 0;i < data_double.size();i++){
          data_as_string << data_double[i];
          if(i < data_double.size() - 1){
            data_as_string << " ";
          }
        }
        return data_as_string.str();
      case(SPUD_INT):
        for(unsigned int i = 0;i < data_int.size();i++){
          data_as_string << data_int[i];
          if(i < data_int.size() - 1){
            data_as_string << " ";
          }
        }
        return data_as_string.str();
      case(SPUD_NONE):
        return "";
      case(SPUD_STRING):
        return data_string;
      default:
        cerr << "SPUD ERROR: Invalid option type" << endl;
        exit(-1);
    }
  }

  // END OF OptionManager::Option CLASS METHODS

  // The option manager
  OptionManager OptionManager::manager;

}
