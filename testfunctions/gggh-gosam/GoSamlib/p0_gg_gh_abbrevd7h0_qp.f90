module     p0_gg_gh_abbrevd7h0_qp
   use p0_gg_gh_config, only: ki => ki_qp
   use p0_gg_gh_kinematics_qp, only: epstensor
   use p0_gg_gh_globalsh0_qp
   implicit none
   private
   complex(ki), dimension(12), public :: abb7
   complex(ki), public :: R2d7
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p0_gg_gh_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_model_qp
      use p0_gg_gh_color_qp, only: TR
      use p0_gg_gh_globalsl1_qp, only: epspow
      implicit none
      abb7(1)=sqrt(mT**2)
      abb7(2)=sqrt2**(-1)
      abb7(3)=spbk2k1**(-1)
      abb7(4)=spak2k3**(-1)
      abb7(5)=spbk3k2**(-1)
      abb7(6)=spak1k2**2*es12
      abb7(7)=abb7(1)*spak1k2
      abb7(8)=abb7(7)**2
      abb7(6)=abb7(6)-2.0_ki*abb7(8)
      abb7(8)=2.0_ki*abb7(1)
      abb7(9)=abb7(2)*abb7(3)*abb7(4)*gHT*e*i_
      abb7(8)=abb7(8)*abb7(9)
      abb7(10)=c2-c1
      abb7(6)=-abb7(8)*abb7(6)*spbk3k1*abb7(10)
      abb7(11)=abb7(1)**2
      abb7(11)=es12-4.0_ki*abb7(11)
      abb7(8)=abb7(10)*abb7(11)*spak1k2*abb7(8)
      abb7(11)=8.0_ki*abb7(9)
      abb7(7)=abb7(11)*abb7(7)
      abb7(12)=abb7(10)*abb7(7)
      abb7(10)=-abb7(5)*abb7(10)
      abb7(7)=-abb7(7)*spbk3k1*abb7(10)
      abb7(10)=abb7(10)*abb7(1)
      abb7(9)=-4.0_ki*abb7(9)*es12*abb7(10)
      abb7(10)=abb7(11)*abb7(10)
      R2d7=0.0_ki
      rat2 = rat2 + R2d7
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='7' value='", &
          & R2d7, "'/>"
      end if
   end subroutine
end module p0_gg_gh_abbrevd7h0_qp
