/* Generated by gliss-attr ($(date)) copyright (c) 2009 IRIT - UPS */

#include <$(proc)/api.h>
#include <$(proc)/id.h>
#include <$(proc)/macros.h>
#include <$(proc)/grt.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int otawa_delayed_t;
typedef otawa_delayed_t (*delayed_fun_t)($(proc)_inst_t *inst);

/*** function definition ***/

static otawa_delayed_t otawa_delayed_UNKNOWN($(proc)_inst_t *inst) {
	return 0;
}

$(foreach instructions)
static otawa_delayed_t otawa_delayed_$(IDENT)($(proc)_inst_t *inst) {
$(otawa_delayed!)
};

$(end)


/*** function table ***/
static delayed_fun_t delayed_funs[] = {
	otawa_delayed_UNKNOWN$(foreach instructions),
	otawa_delayed_$(IDENT)$(end)
};

/**
 * Get the OTAWA kind of the instruction.
 * @return OTAWA kind.
 */
otawa_delayed_t $(proc)_delayed($(proc)_inst_t *inst) {
	return delayed_funs[inst->ident](inst);
}

#ifdef __cplusplus
}
#endif
