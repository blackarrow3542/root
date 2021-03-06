# Module.mk for gviz module
# Copyright (c) 2009 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 2/9/2009

MODNAME      := gviz
MODDIR       := $(ROOT_SRCDIR)/graf2d/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

GVIZDIR      := $(MODDIR)
GVIZDIRS     := $(GVIZDIR)/src
GVIZDIRI     := $(GVIZDIR)/inc

##### libGviz #####
GVIZL        := $(MODDIRI)/LinkDef.h
GVIZDS       := $(call stripsrc,$(MODDIRS)/G__Gviz.cxx)
GVIZDO       := $(GVIZDS:.cxx=.o)
GVIZDH       := $(GVIZDS:.cxx=.h)

GVIZH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
GVIZS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
GVIZO        := $(call stripsrc,$(GVIZS:.cxx=.o))

GVIZDEP      := $(GVIZO:.o=.d) $(GVIZDO:.o=.d)

GVIZLIB      := $(LPATH)/libGviz.$(SOEXT)
GVIZMAP      := $(GVIZLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
GVIZH_REL   := $(patsubst $(MODDIRI)/%.h,include/%.h,$(GVIZH))
ALLHDRS     += $(GVIZH_REL)
ALLLIBS     += $(GVIZLIB)
ALLMAPS     += $(GVIZMAP)
ifeq ($(CXXMODULES),yes)
  CXXMODULES_HEADERS := $(patsubst include/%,header \"%\"\\n,$(GVIZH_REL))
  CXXMODULES_MODULEMAP_CONTENTS += module Graf2d_$(MODNAME) { \\n
  CXXMODULES_MODULEMAP_CONTENTS += $(CXXMODULES_HEADERS)
  CXXMODULES_MODULEMAP_CONTENTS += "export \* \\n"
  CXXMODULES_MODULEMAP_CONTENTS += link \"$(GVIZLIB)\" \\n
  CXXMODULES_MODULEMAP_CONTENTS += } \\n
endif

# include all dependency files
INCLUDEFILES += $(GVIZDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(GVIZDIRI)/%.h
		cp $< $@

$(GVIZLIB):     $(GVIZO) $(GVIZDO) $(ORDER_) $(MAINLIBS) $(GVIZLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGviz.$(SOEXT) $@ "$(GVIZO) $(GVIZDO)" \
		   "$(GVIZLIBEXTRA) $(GRAPHVIZLIB)"

$(call pcmrule,GVIZ)
	$(noop)

$(GVIZDS):      $(GVIZH) $(GVIZL) $(ROOTCLINGEXE) $(call pcmdep,GVIZ)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,GVIZ) -c -writeEmptyRootPCM $(GRAPHVIZINCDIR:%=-I%) $(GVIZH) $(GVIZL)

$(GVIZMAP):     $(GVIZH) $(GVIZL) $(ROOTCLINGEXE) $(call pcmdep,GVIZ)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(GVIZDS) $(call dictModule,GVIZ) -c $(GRAPHVIZINCDIR:%=-I%) $(GVIZH) $(GVIZL)

all-$(MODNAME): $(GVIZLIB)

clean-$(MODNAME):
		@rm -f $(GVIZO) $(GVIZDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(GVIZDEP) $(GVIZDS) $(GVIZDH) $(GVIZLIB) $(GVIZMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(GVIZO) $(GVIZDO): CXXFLAGS += $(GRAPHVIZINCDIR:%=-I%) $(GRAPHVIZCFLAGS)
