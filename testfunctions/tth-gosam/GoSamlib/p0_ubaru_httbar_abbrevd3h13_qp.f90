module     p0_ubaru_httbar_abbrevd3h13_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh13_qp
   implicit none
   private
   complex(ki), dimension(41), public :: abb3
   complex(ki), public :: R2d3
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
      abb3(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb3(2)=NC**(-1)
      abb3(3)=es12**(-1)
      abb3(4)=spak2l3**(-1)
      abb3(5)=spbl3k2**(-1)
      abb3(6)=spak2l4**(-1)
      abb3(7)=spak2l5**(-1)
      abb3(8)=sqrt(mT**2)
      abb3(9)=abb3(2)**2
      abb3(10)=abb3(9)*abb3(8)
      abb3(11)=i_*e*gHT*abb3(1)*TR**2*gs**4
      abb3(12)=abb3(11)*abb3(3)
      abb3(13)=abb3(10)*abb3(12)
      abb3(14)=abb3(13)*mT
      abb3(15)=mT*abb3(2)
      abb3(16)=abb3(15)**2
      abb3(17)=abb3(16)*abb3(12)
      abb3(14)=abb3(14)+abb3(17)
      abb3(14)=abb3(14)*c1
      abb3(17)=abb3(3)*abb3(2)
      abb3(18)=abb3(17)*abb3(11)
      abb3(19)=mT**2
      abb3(20)=abb3(19)*abb3(18)
      abb3(21)=abb3(12)*abb3(15)
      abb3(22)=abb3(21)*abb3(8)
      abb3(20)=abb3(20)+abb3(22)
      abb3(20)=abb3(20)*c2
      abb3(14)=abb3(14)-abb3(20)
      abb3(20)=spbl4k2*spak1k2
      abb3(23)=abb3(20)*abb3(7)
      abb3(24)=abb3(23)*abb3(14)
      abb3(25)=abb3(12)*c1
      abb3(26)=abb3(25)*abb3(9)
      abb3(18)=abb3(18)*c2
      abb3(27)=abb3(26)-abb3(18)
      abb3(28)=abb3(27)*abb3(20)
      abb3(29)=spbl5k2*mH**2*abb3(5)*abb3(4)
      abb3(30)=abb3(28)*abb3(29)
      abb3(24)=abb3(30)+abb3(24)
      abb3(30)=abb3(14)*abb3(6)
      abb3(31)=spbl5k2*spak1k2
      abb3(32)=-abb3(31)*abb3(30)
      abb3(33)=abb3(16)*abb3(25)
      abb3(34)=abb3(18)*abb3(19)
      abb3(33)=abb3(33)-abb3(34)
      abb3(34)=abb3(6)*spak1k2
      abb3(35)=spak2l3*abb3(7)
      abb3(36)=abb3(35)*spbl3k2
      abb3(37)=abb3(34)*abb3(36)
      abb3(38)=-abb3(33)*abb3(37)
      abb3(39)=spak1l3*spbl5l3
      abb3(40)=abb3(39)*spbl4k2
      abb3(41)=-abb3(27)*abb3(40)
      abb3(32)=abb3(41)+abb3(38)+abb3(32)-abb3(24)
      abb3(31)=abb3(6)*abb3(31)
      abb3(23)=abb3(23)+abb3(31)
      abb3(31)=abb3(8)**2
      abb3(23)=-abb3(23)*abb3(31)*abb3(14)
      abb3(26)=abb3(26)*abb3(31)
      abb3(18)=abb3(31)*abb3(18)
      abb3(26)=abb3(26)-abb3(18)
      abb3(20)=-abb3(20)*abb3(29)
      abb3(20)=-abb3(40)+abb3(20)
      abb3(20)=abb3(26)*abb3(20)
      abb3(18)=abb3(19)*abb3(18)
      abb3(26)=abb3(15)*abb3(8)
      abb3(26)=-abb3(25)*abb3(26)**2
      abb3(18)=abb3(18)+abb3(26)
      abb3(18)=abb3(18)*abb3(37)
      abb3(18)=abb3(18)+abb3(23)+abb3(20)
      abb3(18)=2.0_ki*abb3(18)
      abb3(10)=abb3(25)*abb3(10)*mT
      abb3(20)=abb3(22)*c2
      abb3(10)=abb3(10)-abb3(20)
      abb3(20)=2.0_ki*abb3(10)
      abb3(22)=-abb3(6)*abb3(39)*abb3(20)
      abb3(22)=abb3(22)+abb3(24)
      abb3(22)=4.0_ki*abb3(22)
      abb3(23)=abb3(34)*abb3(20)
      abb3(23)=abb3(28)+abb3(23)
      abb3(24)=2.0_ki*spbl5l3
      abb3(23)=abb3(23)*abb3(24)
      abb3(25)=abb3(11)*abb3(8)
      abb3(17)=abb3(25)*abb3(17)
      abb3(17)=abb3(17)+abb3(21)
      abb3(17)=abb3(17)*c2
      abb3(9)=abb3(9)*mT
      abb3(12)=abb3(12)*abb3(9)
      abb3(12)=abb3(12)+abb3(13)
      abb3(12)=abb3(12)*c1
      abb3(12)=abb3(17)-abb3(12)
      abb3(12)=abb3(8)*abb3(12)
      abb3(13)=spbl4k2*abb3(12)
      abb3(17)=abb3(11)*abb3(19)*abb3(2)
      abb3(15)=abb3(15)*abb3(25)
      abb3(15)=abb3(17)+abb3(15)
      abb3(15)=c2*abb3(15)
      abb3(11)=abb3(16)*abb3(11)
      abb3(9)=-abb3(25)*abb3(9)
      abb3(9)=-abb3(11)+abb3(9)
      abb3(9)=c1*abb3(9)
      abb3(9)=abb3(15)+abb3(9)
      abb3(9)=abb3(6)*abb3(9)
      abb3(9)=2.0_ki*abb3(13)+abb3(9)
      abb3(9)=2.0_ki*abb3(9)
      abb3(13)=-4.0_ki*abb3(30)
      abb3(12)=-spbl5k2*abb3(12)
      abb3(10)=abb3(10)*abb3(36)
      abb3(10)=abb3(12)+abb3(10)
      abb3(10)=4.0_ki*abb3(10)
      abb3(12)=-abb3(7)*abb3(14)
      abb3(14)=-abb3(27)*abb3(29)
      abb3(12)=abb3(12)+abb3(14)
      abb3(12)=4.0_ki*abb3(12)
      abb3(14)=-abb3(27)*abb3(24)
      abb3(15)=-spbl4k2*abb3(20)
      abb3(16)=c2*abb3(17)
      abb3(11)=-c1*abb3(11)
      abb3(11)=abb3(16)+abb3(11)
      abb3(11)=abb3(6)*abb3(11)
      abb3(11)=abb3(15)+abb3(11)
      abb3(11)=2.0_ki*abb3(11)*abb3(35)
      abb3(15)=-4.0_ki*abb3(6)*abb3(33)*abb3(35)
      R2d3=abb3(32)
      rat2 = rat2 + R2d3
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='3' value='", &
          & R2d3, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd3h13_qp
