module     p0_ubaru_httbar_abbrevd71h8_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh8_qp
   implicit none
   private
   complex(ki), dimension(0), public :: abb71
   complex(ki), public :: R2d71
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p0_ubaru_httbar_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_color_qp, only: TR
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      implicit none
      R2d71=0.0_ki
      rat2 = rat2 + R2d71
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='71' value='", &
          & R2d71, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd71h8_qp
