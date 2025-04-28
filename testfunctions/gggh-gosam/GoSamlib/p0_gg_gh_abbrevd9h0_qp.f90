module     p0_gg_gh_abbrevd9h0_qp
   use p0_gg_gh_config, only: ki => ki_qp
   use p0_gg_gh_kinematics_qp, only: epstensor
   use p0_gg_gh_globalsh0_qp
   implicit none
   private
   complex(ki), dimension(15), public :: abb9
   complex(ki), public :: R2d9
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
      abb9(1)=sqrt(mT**2)
      abb9(2)=sqrt2**(-1)
      abb9(3)=spbk2k1**(-1)
      abb9(4)=spak2k3**(-1)
      abb9(5)=spbk3k2**(-1)
      abb9(6)=spak2l4**(-1)
      abb9(7)=spbl4k2**(-1)
      abb9(8)=c1-c2
      abb9(9)=i_*e*gHT*abb9(4)*abb9(3)*abb9(2)
      abb9(10)=abb9(8)*abb9(9)*abb9(5)*abb9(1)
      abb9(11)=-abb9(10)*spak1l4*spbl4k3
      abb9(8)=-abb9(9)*abb9(8)
      abb9(9)=abb9(8)*spak1k2
      abb9(12)=-abb9(1)*abb9(9)
      abb9(13)=-abb9(7)*abb9(6)*abb9(12)*mH**2
      abb9(13)=abb9(13)-abb9(11)
      abb9(13)=es12*abb9(13)
      abb9(9)=-abb9(1)**3*abb9(9)
      abb9(9)=2.0_ki*abb9(9)+abb9(13)
      abb9(9)=2.0_ki*abb9(9)
      abb9(11)=-4.0_ki*abb9(11)
      abb9(12)=4.0_ki*abb9(12)
      abb9(13)=spak1k2*abb9(10)
      abb9(13)=2.0_ki*abb9(13)
      abb9(14)=-spbl4k3*abb9(13)
      abb9(8)=2.0_ki*spak2l4*abb9(8)*abb9(1)
      abb9(15)=-spbl4k2*spak2l4
      abb9(15)=es12+abb9(15)
      abb9(15)=-2.0_ki*abb9(10)*abb9(15)
      abb9(13)=-spbk3k1*abb9(13)
      abb9(10)=8.0_ki*abb9(10)
      R2d9=0.0_ki
      rat2 = rat2 + R2d9
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='9' value='", &
          & R2d9, "'/>"
      end if
   end subroutine
end module p0_gg_gh_abbrevd9h0_qp
