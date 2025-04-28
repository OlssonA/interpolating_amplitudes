module     p0_ubaru_httbar_abbrevd72h9_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh9_qp
   implicit none
   private
   complex(ki), dimension(40), public :: abb72
   complex(ki), public :: R2d72
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
      abb72(1)=1.0_ki/(-mT**2+es34)
      abb72(2)=NC**(-1)
      abb72(3)=es12**(-1)
      abb72(4)=spbl4k2**(-1)
      abb72(5)=spak2l5**(-1)
      abb72(6)=sqrt(mT**2)
      abb72(7)=spak2l3**(-1)
      abb72(8)=spbl3k2**(-1)
      abb72(9)=spak2l4**(-1)
      abb72(10)=abb72(2)*c1
      abb72(11)=abb72(10)-c2
      abb72(12)=abb72(11)*abb72(2)
      abb72(13)=abb72(12)-c1
      abb72(14)=spbl3k2*abb72(4)
      abb72(15)=abb72(14)*spak1l3
      abb72(16)=abb72(15)+spak1l4
      abb72(16)=abb72(16)*abb72(13)
      abb72(17)=spak1l4*NC
      abb72(18)=abb72(15)*NC
      abb72(17)=abb72(17)+abb72(18)
      abb72(17)=abb72(17)*c2
      abb72(16)=abb72(17)+abb72(16)
      abb72(16)=abb72(16)*mT
      abb72(17)=-abb72(2)*spak1l4*abb72(11)
      abb72(19)=c2*NC
      abb72(20)=abb72(19)-c1
      abb72(21)=-spak1l4*abb72(20)
      abb72(17)=abb72(17)+abb72(21)
      abb72(21)=abb72(17)*abb72(6)
      abb72(16)=abb72(16)-abb72(21)
      abb72(16)=abb72(16)*spbl5k2
      abb72(12)=abb72(12)+abb72(20)
      abb72(22)=spak1k2*abb72(5)
      abb72(23)=-abb72(12)*abb72(22)*spbl3k2*spal3l4
      abb72(24)=abb72(23)*mT
      abb72(16)=abb72(16)+abb72(24)
      abb72(25)=TR**2*abb72(1)*gs**4*abb72(3)*gHT*e*i_
      abb72(26)=2.0_ki*abb72(25)
      abb72(27)=-abb72(16)*abb72(26)
      abb72(28)=spal3l4*spbl4l3
      abb72(29)=mH**2
      abb72(30)=abb72(29)*abb72(7)
      abb72(31)=abb72(30)*abb72(8)
      abb72(32)=abb72(31)*spak2l4
      abb72(33)=abb72(32)*spbl4k2
      abb72(28)=abb72(28)+abb72(33)
      abb72(28)=abb72(28)*spak1l4
      abb72(33)=spak1k2*spal3l4
      abb72(34)=abb72(30)*abb72(33)
      abb72(35)=abb72(30)*spak2l4
      abb72(36)=abb72(35)*spak1l3
      abb72(28)=abb72(28)-abb72(34)+abb72(36)
      abb72(34)=-abb72(28)*abb72(13)
      abb72(28)=-c2*NC*abb72(28)
      abb72(36)=abb72(6)**2
      abb72(37)=2.0_ki*abb72(36)
      abb72(38)=-abb72(17)*abb72(37)
      abb72(28)=abb72(38)+abb72(28)+abb72(34)
      abb72(28)=abb72(6)*abb72(28)
      abb72(34)=mT*abb72(6)
      abb72(38)=abb72(4)*abb72(9)
      abb72(33)=abb72(34)*abb72(12)*abb72(38)*spbl3k2*abb72(33)
      abb72(39)=2.0_ki*spak1l4
      abb72(40)=abb72(39)+abb72(15)
      abb72(40)=abb72(40)*abb72(13)
      abb72(39)=NC*abb72(39)
      abb72(18)=abb72(39)+abb72(18)
      abb72(18)=c2*abb72(18)
      abb72(18)=abb72(18)+abb72(40)
      abb72(18)=abb72(18)*abb72(36)
      abb72(18)=abb72(18)+abb72(33)
      abb72(18)=mT*abb72(18)
      abb72(18)=abb72(28)+abb72(18)
      abb72(18)=spbl5k2*abb72(18)
      abb72(28)=abb72(36)+abb72(34)
      abb72(24)=abb72(28)*abb72(24)
      abb72(18)=abb72(24)+abb72(18)
      abb72(18)=abb72(18)*abb72(26)
      abb72(24)=abb72(25)*spbl5k2
      abb72(21)=8.0_ki*abb72(24)*abb72(21)
      abb72(15)=spbl5k2*abb72(15)*abb72(12)
      abb72(15)=abb72(23)+abb72(15)
      abb72(23)=4.0_ki*abb72(25)
      abb72(15)=abb72(23)*mT*abb72(15)
      abb72(13)=abb72(19)+abb72(13)
      abb72(19)=spak1l3*abb72(30)*abb72(4)
      abb72(25)=abb72(31)*spak1l4
      abb72(19)=abb72(19)+abb72(25)
      abb72(13)=4.0_ki*mT*abb72(24)*abb72(19)*abb72(13)
      abb72(16)=abb72(16)*abb72(23)
      abb72(19)=abb72(26)*spbl5k2
      abb72(23)=abb72(19)*abb72(6)
      abb72(24)=-abb72(23)*spak1l3*abb72(12)
      abb72(25)=2.0_ki*abb72(34)
      abb72(28)=-abb72(29)-abb72(25)+abb72(37)
      abb72(11)=-abb72(2)*abb72(22)*abb72(11)
      abb72(20)=-abb72(22)*abb72(20)
      abb72(11)=abb72(11)+abb72(20)
      abb72(20)=mT*abb72(11)*abb72(28)
      abb72(10)=abb72(10)*spak1k2
      abb72(22)=c2*spak1k2
      abb72(10)=abb72(10)-abb72(22)
      abb72(10)=abb72(10)*abb72(2)
      abb72(22)=abb72(22)*NC
      abb72(28)=c1*spak1k2
      abb72(10)=abb72(10)+abb72(22)-abb72(28)
      abb72(22)=-abb72(31)*abb72(10)
      abb72(10)=-mT**2*abb72(38)*abb72(10)
      abb72(10)=2.0_ki*abb72(10)+abb72(22)
      abb72(10)=spbl5k2*abb72(6)*abb72(10)
      abb72(10)=abb72(20)+abb72(10)
      abb72(10)=abb72(10)*abb72(26)
      abb72(20)=-mT*abb72(19)*abb72(14)*abb72(17)
      abb72(22)=abb72(35)*abb72(11)
      abb72(11)=-abb72(25)*abb72(14)*abb72(11)
      abb72(14)=-spbl5k2*abb72(17)*abb72(4)*spbl4l3
      abb72(11)=abb72(14)+abb72(11)+abb72(22)
      abb72(11)=abb72(26)*mT*abb72(11)
      abb72(14)=abb72(23)*spal3l4*abb72(12)
      abb72(17)=abb72(6)*abb72(32)*abb72(12)
      abb72(12)=-mT*abb72(37)*abb72(4)*abb72(12)
      abb72(12)=abb72(17)+abb72(12)
      abb72(12)=abb72(12)*abb72(19)
      R2d72=abb72(27)
      rat2 = rat2 + R2d72
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='72' value='", &
          & R2d72, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd72h9_qp
