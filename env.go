package metaverse

import (
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	gym "github.com/unixpickle/gym-socket-api/binding-go"
)

// Env is a Universe environment implementing anyrl.Env.
type Env struct {
	GymEnv      gym.Env
	Imager      Imager
	ActionSpace *ActionSpace

	// Mask is used to prevent the agent from making some
	// of the technically allowed actions.
	//
	// If nil, no mask is used.
	Mask *ActionSpace
}

// Reset resets the environment.
func (e *Env) Reset() (obs anyvec.Vector, err error) {
	defer essentials.AddCtxTo("reset Universe environment", &err)
	rawObs, err := e.GymEnv.Reset()
	if err != nil {
		return
	}
	obs = e.Imager.Image(rawObs)
	return
}

// Step takes a step in the environment.
func (e *Env) Step(actionVec anyvec.Vector) (obs anyvec.Vector, reward float64,
	done bool, err error) {
	actionObj := e.ActionSpace.VecToObj(actionVec)
	if e.Mask != nil {
		actionObj = e.Mask.Mask(actionObj)
	}
	rawObs, reward, done, _, err := e.GymEnv.Step(actionObj)
	if err != nil {
		return
	}
	obs = e.Imager.Image(rawObs)
	return
}
