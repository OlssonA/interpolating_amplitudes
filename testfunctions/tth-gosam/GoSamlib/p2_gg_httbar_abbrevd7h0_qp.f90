module     p2_gg_httbar_abbrevd7h0_qp
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_kinematics_qp, only: epstensor
   use p2_gg_httbar_globalsh0_qp
   implicit none
   private
   complex(ki), dimension(5), public :: abb7
   complex(ki), public :: R2d7
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p2_gg_httbar_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_color_qp, only: TR
      use p2_gg_httbar_globalsl1_qp, only: epspow
      implicit none
      abb7(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb7(2)=es12**(-1)
      abb7(3)=1.0_ki/(-mT**2+es34)
      abb7(4)=abb7(3)*spal3l4*spak2l5
      abb7(5)=abb7(1)*spal3l5*spak2l4
      abb7(4)=abb7(4)+abb7(5)
      abb7(5)=c2-c1
      abb7(4)=9.0_ki/8.0_ki*abb7(2)*gHT*e*spbe2e1*spae1e2*spbl3k2*NC*TR*i_*gs**&
      &4*abb7(5)*abb7(4)
      R2d7=0.0_ki
      rat2 = rat2 + R2d7
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='7' value='", &
          & R2d7, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd7h0_qp
