  interface addref
     module procedure addref_REFCOUNT_TYPE
  end interface

  interface incref
     module procedure incref_REFCOUNT_TYPE
  end interface

  interface decref
     module procedure decref_REFCOUNT_TYPE
  end interface

  interface has_references
     module procedure has_references_REFCOUNT_TYPE
  end interface

