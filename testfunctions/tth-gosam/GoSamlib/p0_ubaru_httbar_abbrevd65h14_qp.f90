module     p0_ubaru_httbar_abbrevd65h14_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh14_qp
   implicit none
   private
   complex(ki), dimension(23), public :: abb65
   complex(ki), public :: R2d65
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
      abb65(1)=1.0_ki/(-mT**2+es34)
      abb65(2)=NC**(-1)
      abb65(3)=spak2l3**(-1)
      abb65(4)=spbl3k2**(-1)
      abb65(5)=spak2l4**(-1)
      abb65(6)=sqrt(mT**2)
      abb65(7)=i_*e*gHT*abb65(1)*TR**2*gs**4
      abb65(8)=abb65(7)*c2
      abb65(9)=abb65(8)*abb65(2)
      abb65(10)=2.0_ki*abb65(9)
      abb65(7)=abb65(7)*abb65(2)**2
      abb65(11)=abb65(7)*c1
      abb65(11)=abb65(10)-abb65(11)
      abb65(12)=abb65(8)*NC
      abb65(12)=abb65(12)-abb65(11)
      abb65(13)=spbl5k1*spak2l5
      abb65(14)=abb65(12)*abb65(13)
      abb65(15)=2.0_ki*spbl4l3
      abb65(16)=abb65(14)*abb65(15)
      abb65(17)=mT**2
      abb65(18)=abb65(17)*abb65(5)
      abb65(19)=abb65(18)*abb65(8)
      abb65(20)=abb65(8)*mT
      abb65(21)=abb65(6)*abb65(5)
      abb65(22)=abb65(20)*abb65(21)
      abb65(19)=abb65(19)+abb65(22)
      abb65(19)=abb65(19)*NC
      abb65(7)=abb65(5)*abb65(7)
      abb65(17)=abb65(7)*abb65(17)
      abb65(7)=mT*abb65(7)*abb65(6)
      abb65(17)=abb65(17)+abb65(7)
      abb65(17)=abb65(17)*c1
      abb65(17)=abb65(17)+abb65(19)
      abb65(19)=abb65(13)*abb65(17)
      abb65(23)=abb65(6)+mT
      abb65(11)=-abb65(23)*abb65(11)
      abb65(8)=abb65(6)*abb65(8)
      abb65(8)=abb65(20)+abb65(8)
      abb65(8)=NC*abb65(8)
      abb65(8)=abb65(8)+abb65(11)
      abb65(8)=spbl4k1*abb65(6)*abb65(8)
      abb65(11)=abb65(21)*mT
      abb65(10)=-abb65(11)*abb65(10)
      abb65(20)=NC*abb65(22)
      abb65(7)=c1*abb65(7)
      abb65(7)=abb65(20)+abb65(10)+abb65(7)
      abb65(7)=spbl3k1*spak2l3*abb65(7)
      abb65(10)=abb65(18)+abb65(11)
      abb65(9)=abb65(9)*abb65(10)
      abb65(10)=2.0_ki*abb65(13)
      abb65(10)=-abb65(9)*abb65(10)
      abb65(11)=abb65(3)*mH**2*spbl4k2*abb65(4)
      abb65(13)=abb65(14)*abb65(11)
      abb65(7)=abb65(7)+abb65(8)+abb65(13)+abb65(10)+abb65(19)
      abb65(7)=2.0_ki*abb65(7)
      abb65(8)=abb65(12)*abb65(15)
      abb65(10)=abb65(12)*abb65(11)
      abb65(9)=abb65(10)-2.0_ki*abb65(9)+abb65(17)
      abb65(9)=2.0_ki*abb65(9)
      R2d65=0.0_ki
      rat2 = rat2 + R2d65
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='65' value='", &
          & R2d65, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd65h14_qp
