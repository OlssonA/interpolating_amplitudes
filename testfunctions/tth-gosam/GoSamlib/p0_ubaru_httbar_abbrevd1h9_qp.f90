module     p0_ubaru_httbar_abbrevd1h9_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh9_qp
   implicit none
   private
   complex(ki), dimension(33), public :: abb1
   complex(ki), public :: R2d1
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
      abb1(1)=1.0_ki/(-mT**2+es34)
      abb1(2)=NC**(-1)
      abb1(3)=es12**(-1)
      abb1(4)=spbl4k2**(-1)
      abb1(5)=spak2l5**(-1)
      abb1(6)=sqrt(mT**2)
      abb1(7)=spak2l3**(-1)
      abb1(8)=spbl3k2**(-1)
      abb1(9)=i_*e*gHT*abb1(1)*TR**2*gs**4
      abb1(10)=abb1(9)*abb1(3)
      abb1(11)=abb1(10)*abb1(6)
      abb1(12)=abb1(2)**2
      abb1(13)=abb1(11)*abb1(12)
      abb1(14)=abb1(12)*abb1(9)
      abb1(15)=abb1(3)*abb1(14)*mT
      abb1(13)=abb1(13)+abb1(15)
      abb1(13)=abb1(13)*c1
      abb1(16)=mT*abb1(2)
      abb1(17)=abb1(10)*abb1(16)
      abb1(18)=abb1(11)*abb1(2)
      abb1(19)=abb1(18)+abb1(17)
      abb1(19)=abb1(19)*c2
      abb1(13)=abb1(13)-abb1(19)
      abb1(19)=abb1(13)*spbl5k2
      abb1(20)=abb1(19)*spak1l4
      abb1(21)=abb1(15)*c1
      abb1(17)=abb1(17)*c2
      abb1(17)=abb1(21)-abb1(17)
      abb1(21)=abb1(17)*abb1(5)
      abb1(22)=spal3l4*spbl3k2
      abb1(23)=abb1(22)*spak1k2
      abb1(24)=abb1(21)*abb1(23)
      abb1(25)=spbl3k2*abb1(4)
      abb1(26)=abb1(25)*spak1l3
      abb1(27)=abb1(26)*spbl5k2
      abb1(28)=-abb1(17)*abb1(27)
      abb1(24)=abb1(28)+abb1(24)-abb1(20)
      abb1(23)=abb1(5)*abb1(23)
      abb1(23)=-abb1(27)+abb1(23)
      abb1(23)=abb1(23)*abb1(17)
      abb1(20)=-abb1(20)+abb1(23)
      abb1(20)=2.0_ki*abb1(20)*abb1(6)**2
      abb1(23)=abb1(11)*abb1(16)
      abb1(27)=abb1(10)*abb1(2)
      abb1(28)=mT**2
      abb1(29)=abb1(27)*abb1(28)
      abb1(23)=abb1(23)+abb1(29)
      abb1(23)=abb1(23)*c2
      abb1(15)=abb1(15)*abb1(6)
      abb1(10)=abb1(10)*abb1(12)
      abb1(29)=abb1(10)*abb1(28)
      abb1(15)=abb1(15)+abb1(29)
      abb1(15)=abb1(15)*c1
      abb1(15)=abb1(23)-abb1(15)
      abb1(15)=abb1(6)*abb1(15)
      abb1(23)=spak1l4*abb1(15)
      abb1(29)=abb1(28)*abb1(11)
      abb1(30)=abb1(12)*c1
      abb1(31)=abb1(29)*abb1(30)
      abb1(32)=abb1(18)*c2
      abb1(33)=abb1(32)*abb1(28)
      abb1(31)=abb1(31)-abb1(33)
      abb1(26)=-abb1(31)*abb1(26)
      abb1(23)=abb1(23)+abb1(26)
      abb1(23)=8.0_ki*abb1(5)*abb1(23)
      abb1(26)=2.0_ki*abb1(5)
      abb1(33)=-abb1(15)*abb1(26)
      abb1(19)=abb1(33)+abb1(19)
      abb1(33)=2.0_ki*spak1k2
      abb1(19)=abb1(19)*abb1(33)
      abb1(33)=abb1(17)*spbl5k2
      abb1(26)=abb1(31)*abb1(26)
      abb1(26)=abb1(26)+abb1(33)
      abb1(25)=2.0_ki*abb1(25)
      abb1(26)=spak1k2*abb1(26)*abb1(25)
      abb1(11)=abb1(11)*abb1(30)
      abb1(11)=abb1(32)-abb1(11)
      abb1(22)=4.0_ki*abb1(11)*abb1(22)
      abb1(13)=-2.0_ki*abb1(13)
      abb1(17)=-abb1(17)*abb1(25)
      abb1(16)=c2*abb1(16)*abb1(9)
      abb1(14)=abb1(14)*c1
      abb1(25)=mT*abb1(14)
      abb1(16)=abb1(16)-abb1(25)
      abb1(16)=abb1(16)*abb1(5)
      abb1(25)=2.0_ki*spbl5k2
      abb1(11)=abb1(11)*abb1(25)
      abb1(11)=abb1(16)+abb1(11)
      abb1(16)=-2.0_ki*spal3l4*abb1(11)
      abb1(30)=4.0_ki*spal3l4*abb1(21)
      abb1(15)=-abb1(15)*abb1(25)
      abb1(25)=abb1(28)*abb1(6)
      abb1(31)=mT**3
      abb1(25)=abb1(25)+abb1(31)
      abb1(14)=abb1(25)*abb1(14)
      abb1(9)=-c2*abb1(25)*abb1(9)*abb1(2)
      abb1(9)=abb1(9)+abb1(14)
      abb1(9)=abb1(5)*abb1(9)
      abb1(9)=abb1(9)+abb1(15)
      abb1(9)=abb1(4)*abb1(9)
      abb1(14)=abb1(7)*abb1(8)*spak2l4*mH**2
      abb1(11)=-abb1(11)*abb1(14)
      abb1(9)=abb1(9)+abb1(11)
      abb1(9)=2.0_ki*abb1(9)
      abb1(11)=abb1(18)*abb1(28)
      abb1(15)=-abb1(31)*abb1(27)
      abb1(11)=abb1(15)-abb1(11)
      abb1(11)=c2*abb1(11)
      abb1(10)=abb1(31)*abb1(10)
      abb1(12)=abb1(12)*abb1(29)
      abb1(10)=abb1(10)+abb1(12)
      abb1(10)=c1*abb1(10)
      abb1(10)=abb1(11)+abb1(10)
      abb1(10)=abb1(4)*abb1(5)*abb1(10)
      abb1(11)=abb1(21)*abb1(14)
      abb1(10)=abb1(10)+abb1(11)
      abb1(10)=4.0_ki*abb1(10)
      R2d1=abb1(24)
      rat2 = rat2 + R2d1
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='1' value='", &
          & R2d1, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd1h9_qp
