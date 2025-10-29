`ifndef REG_MODEL
`define REG_MODEL

class CTRL_REG extends uvm_reg;
  `uvm_object_utils(CTRL_REG)

  //---------------------------------------
  // Constructor
  //---------------------------------------
  function new(string name = "CTRL_REG");
    super.new(name, 32, UVM_NO_COVERAGE);
  endfunction

  //---------------------------------------
    rand uvm_reg_field ENABLE;
    rand uvm_reg_field MODE;
    rand uvm_reg_field INTR_EN;
  //---------------------------------------
  function void build;
    ENABLE = uvm_reg_field::type_id::create("ENABLE");
    ENABLE.configure(.parent(this),
                           .size(1),
                           .lsb_pos(0),
                           .msb_pos(0),
                           .access("RO"),
                           .volatile(0),
                           .reset(0),
                           .has_reset(1),
                           .is_rand(1),
                           .individually_accessible(0));
    MODE = uvm_reg_field::type_id::create("MODE");
    MODE.configure(.parent(this),
                           .size(2),
                           .lsb_pos(1),
                           .msb_pos(2),
                           .access("RO"),
                           .volatile(0),
                           .reset(0),
                           .has_reset(1),
                           .is_rand(1),
                           .individually_accessible(0));
    INTR_EN = uvm_reg_field::type_id::create("INTR_EN");
    INTR_EN.configure(.parent(this),
                           .size(1),
                           .lsb_pos(3),
                           .msb_pos(3),
                           .access("RO"),
                           .volatile(0),
                           .reset(0),
                           .has_reset(1),
                           .is_rand(1),
                           .individually_accessible(0));
  endfunction
endclass

class STATUS_REG extends uvm_reg;
  `uvm_object_utils(STATUS_REG)

  //---------------------------------------
  // Constructor
  //---------------------------------------
  function new(string name = "STATUS_REG");
    super.new(name, 32, UVM_NO_COVERAGE);
  endfunction

  //---------------------------------------
    rand uvm_reg_field BUSY;
    rand uvm_reg_field ERROR;
  //---------------------------------------
  function void build;
    BUSY = uvm_reg_field::type_id::create("BUSY");
    BUSY.configure(.parent(this),
                           .size(1),
                           .lsb_pos(0),
                           .msb_pos(0),
                           .access("RW"),
                           .volatile(0),
                           .reset(0),
                           .has_reset(1),
                           .is_rand(1),
                           .individually_accessible(0));
    ERROR = uvm_reg_field::type_id::create("ERROR");
    ERROR.configure(.parent(this),
                           .size(1),
                           .lsb_pos(1),
                           .msb_pos(1),
                           .access("RW"),
                           .volatile(0),
                           .reset(0),
                           .has_reset(1),
                           .is_rand(1),
                           .individually_accessible(0));
  endfunction
endclass

class DATA_REG extends uvm_reg;
  `uvm_object_utils(DATA_REG)

  //---------------------------------------
  // Constructor
  //---------------------------------------
  function new(string name = "DATA_REG");
    super.new(name, 32, UVM_NO_COVERAGE);
  endfunction

  //---------------------------------------
    rand uvm_reg_field DATA;
    DATA = uvm_reg_field::type_id::create("DATA");
    DATA.configure(.parent(this),
                           .size(1),
                           .lsb_pos(0),
                           .msb_pos(31),
                           .access("RW"),
                           .volatile(0),
                           .reset(0),
                           .has_reset(1),
                           .is_rand(1),
                           .individually_accessible(0));
  endfunction
endclass

//-------------------------------------------------------------------------
//	Register Block Definition
//-------------------------------------------------------------------------
class dma_reg_model extends uvm_reg_block;
  `uvm_object_utils(dma_reg_model)

  //---------------------------------------
  // Register Instances
  //---------------------------------------
  rand CTRL_REG reg_ctrl_reg;
  rand  reg_;
  rand STATUS_REG reg_status_reg;
  rand DATA_REG reg_data_reg;

  //---------------------------------------
  // Constructor
  //---------------------------------------
  function new (string name = "");
    super.new(name, build_coverage(UVM_NO_COVERAGE));
  endfunction

  //---------------------------------------
  // Build Phase
  //---------------------------------------
  function void build();
    reg_ctrl_reg = CTRL_REG::type_id::create("reg_ctrl_reg");
    reg_ctrl_reg.build();
    reg_ctrl_reg.configure(this);
    reg_ = ::type_id::create("reg_");
    reg_.build();
    reg_.configure(this);
    reg_status_reg = STATUS_REG::type_id::create("reg_status_reg");
    reg_status_reg.build();
    reg_status_reg.configure(this);
    reg_data_reg = DATA_REG::type_id::create("reg_data_reg");
    reg_data_reg.build();
    reg_data_reg.configure(this);
    //---------------------------------------
    // Memory Map Creation and Register Map
    //---------------------------------------
    default_map = create_map("my_map", 0, 4, UVM_LITTLE_ENDIAN);
    default_map.add_reg(reg_ctrl_reg, 'h0, "RW");
    default_map.add_reg(reg_, 'h0, "RW");
    default_map.add_reg(reg_status_reg, 'h0, "RW");
    default_map.add_reg(reg_data_reg, 'h0, "RW");
    lock_model();
  endfunction
endclass

`endif // REG_MODEL
