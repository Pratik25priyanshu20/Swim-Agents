# Main agent that orchestrates HOMOGEN, CALIBRO, and VISIOS

class MainAgentController:
    def __init__(self, homogen_agent, calibro_agent, visios_agent):
        self.homogen = homogen_agent
        self.calibro = calibro_agent
        self.visios = visios_agent

    def run_full_pipeline(self):
        print("📥 Starting ingestion with HOMOGEN...")
        self.homogen.run()

        print("🔧 Running calibration with CALIBRO...")
        self.calibro.run()

        print("🖼️ Performing visual interpretation with VISIOS...")
        self.visios.run()
        