module     p0_gg_gh_abbrevd11h1_qp
   use p0_gg_gh_config, only: ki => ki_qp
   use p0_gg_gh_kinematics_qp, only: epstensor
   use p0_gg_gh_globalsh1_qp
   implicit none
   private
   complex(ki), dimension(12), public :: abb11
   complex(ki), public :: R2d11
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
      abb11(1)=sqrt(mT**2)
      abb11(2)=sqrt2**(-1)
      abb11(3)=spak2k3**(-1)
      abb11(4)=spbk3k2**(-1)
      abb11(5)=spak1k2**(-1)
      abb11(6)=c1-c2
      abb11(7)=2.0_ki*spbl4k1
      abb11(8)=gHT*i_*e*abb11(4)*abb11(3)*abb11(2)*abb11(1)
      abb11(9)=abb11(8)*spak2l4
      abb11(10)=abb11(9)*spbk3k1
      abb11(11)=-abb11(7)*abb11(10)*abb11(6)
      abb11(12)=abb11(6)*abb11(5)
      abb11(10)=2.0_ki*abb11(10)*abb11(12)
      abb11(9)=abb11(9)*abb11(12)
      abb11(7)=-abb11(9)*abb11(7)
      abb11(6)=abb11(6)*abb11(8)
      abb11(8)=spbk3k1*abb11(6)
      abb11(9)=-spbl4k3*abb11(9)
      abb11(8)=2.0_ki*abb11(8)+abb11(9)
      abb11(8)=2.0_ki*abb11(8)
      abb11(6)=-8.0_ki*abb11(5)*abb11(6)
      R2d11=0.0_ki
      rat2 = rat2 + R2d11
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='11' value='", &
          & R2d11, "'/>"
      end if
   end subroutine
end module p0_gg_gh_abbrevd11h1_qp
