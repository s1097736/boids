let Scene = {
	w: 600,
	h: 600,
	swarm: [],
	neighbours(x) {
		let r = []
		for (let p of this.swarm) {
			if (dist(p.pos.x, p.pos.y, x.x, x.y) <= 100) {
				r.push(p)
			}
		}
		return r
	}
}

class Particle {

	constructor() {
		this.pos = createVector(random(0, Scene.w), random(0, Scene.h))
		this.dir = p5.Vector.random2D()
	}

	wrap() {
		if (this.pos.x < 0) this.pos.x += Scene.w
		if (this.pos.y < 0) this.pos.y += Scene.h
		if (this.pos.x > Scene.w) this.pos.x -= Scene.w
		if (this.pos.y > Scene.h) this.pos.y -= Scene.h
	}

	draw() {
		fill(0)
		ellipse(this.pos.x, this.pos.y, 5, 5)
	}

	step() {
		let N = Scene.neighbours(this.pos),
			avg_sin = 0, avg_cos = 0,
			avg_p = createVector(0, 0),
			avg_d = createVector(0, 0)

		for (let n of N) {
			avg_sin += Math.sin(n.dir.heading()) / N.length
			avg_cos += Math.cos(n.dir.heading()) / N.length

			avg_p.add(n.pos)

			if (n != this) {
				let away = p5.Vector.sub(this.pos, n.pos)
				away.div(away.magSq())
				avg_d.add(away)
			}
		}
		avg_p.div(N.length)
		avg_d.div(N.length)

		let avg_angle = Math.atan2(avg_cos, avg_sin)
		avg_angle += Math.random() * 0.5 - 0.25
		this.dir = p5.Vector.fromAngle(avg_angle)

		let cohesion = p5.Vector.sub(avg_p, this.pos)
		cohesion.div(100)
		this.dir.add(cohesion)

		avg_d.mult(20)
		this.dir.add(avg_d)
		this.dir.mult(4)

		this.pos.add(this.dir)
		this.wrap()
	}
}

function setup() {
	createCanvas(Scene.w, Scene.h)
	for (let _ of Array(200)) {
		Scene.swarm.push(new Particle())
	}
}

function draw() {
	background(220)
	for (let p of Scene.swarm) {
		p.step()
		p.draw()
	}
}
