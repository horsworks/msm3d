function(msm3d_add_app target source)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE msm3d::core)
  set_target_properties(${target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
endfunction()

msm3d_add_app(cam_demo camera/calibration_demo.cpp)
msm3d_add_app(phase_demo phase/processor_demo.cpp)
msm3d_add_app(iso_demo phase/inspect_iso_phase.cpp)
msm3d_add_app(phase_map_demo phase/inspect_phase_map.cpp)
msm3d_add_app(msm_demo msm/calibration_demo.cpp)
