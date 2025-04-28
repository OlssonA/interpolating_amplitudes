module     p0_ubaru_httbar_abbrevd67h1_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh1_qp
   implicit none
   private
   complex(ki), dimension(18), public :: abb67
   complex(ki), public :: R2d67
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
      abb67(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb67(2)=NC**(-1)
      abb67(3)=spak2l3**(-1)
      abb67(4)=spbl3k2**(-1)
      abb67(5)=spbl5k2**(-1)
      abb67(6)=sqrt(mT**2)
      abb67(7)=i_*e*gHT*abb67(1)*TR**2*gs**4
      abb67(8)=abb67(7)*abb67(2)
      abb67(9)=2.0_ki*c2
      abb67(10)=abb67(8)*abb67(9)
      abb67(11)=abb67(2)**2
      abb67(11)=abb67(11)+1.0_ki
      abb67(7)=abb67(7)*abb67(11)
      abb67(11)=abb67(7)*c1
      abb67(11)=abb67(10)-abb67(11)
      abb67(12)=spak1l4*spbl4k2
      abb67(13)=-abb67(11)*abb67(12)
      abb67(14)=2.0_ki*spal3l5
      abb67(15)=abb67(13)*abb67(14)
      abb67(16)=abb67(5)*mT**2
      abb67(17)=mT*abb67(6)*abb67(5)
      abb67(18)=abb67(16)+abb67(17)
      abb67(8)=abb67(9)*abb67(8)*abb67(18)
      abb67(9)=abb67(16)*c1
      abb67(16)=abb67(17)*c1
      abb67(9)=abb67(9)+abb67(16)
      abb67(9)=abb67(9)*abb67(7)
      abb67(8)=abb67(8)-abb67(9)
      abb67(9)=-abb67(12)*abb67(8)
      abb67(12)=spak2l5*mH**2*abb67(4)*abb67(3)
      abb67(13)=abb67(13)*abb67(12)
      abb67(18)=abb67(6)+mT
      abb67(18)=-spak1l5*abb67(6)*abb67(18)*abb67(11)
      abb67(7)=abb67(7)*abb67(16)
      abb67(10)=-abb67(17)*abb67(10)
      abb67(7)=abb67(7)+abb67(10)
      abb67(7)=spak1l3*spbl3k2*abb67(7)
      abb67(7)=abb67(7)+abb67(18)+abb67(13)+abb67(9)
      abb67(7)=2.0_ki*abb67(7)
      abb67(9)=-abb67(11)*abb67(14)
      abb67(10)=-abb67(11)*abb67(12)
      abb67(8)=abb67(10)-abb67(8)
      abb67(8)=2.0_ki*abb67(8)
      R2d67=0.0_ki
      rat2 = rat2 + R2d67
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='67' value='", &
          & R2d67, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd67h1_qp
