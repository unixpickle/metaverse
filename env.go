package metaverse

import (
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	gym "github.com/unixpickle/gym-socket-api/binding-go"
)

// Env is a Universe environment implementing anyrl.Env.
type Env struct {
	GymEnv      gym.Env
	Imager      *Imager
	ActionSpace *ActionSpace
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
	rawObs, reward, done, _, err := e.GymEnv.Step(actionObj)
	if err != nil {
		return
	}
	obs = e.Imager.Image(rawObs)
	return
}
