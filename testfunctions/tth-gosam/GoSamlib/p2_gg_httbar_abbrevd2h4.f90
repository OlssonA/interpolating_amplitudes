module     p2_gg_httbar_abbrevd2h4
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_kinematics, only: epstensor
   use p2_gg_httbar_globalsh4
   implicit none
   private
   complex(ki), dimension(36), public :: abb2
   complex(ki), public :: R2d2
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p2_gg_httbar_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_model
      use p2_gg_httbar_color, only: TR
      use p2_gg_httbar_globalsl1, only: epspow
      implicit none
      abb2(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb2(2)=sqrt(mT**2)
      abb2(3)=spak2l3**(-1)
      abb2(4)=spbl3k2**(-1)
      abb2(5)=spak2l4**(-1)
      abb2(6)=spbl5k2**(-1)
      abb2(7)=mT**2
      abb2(8)=i_*TR*e*gHT*abb2(1)*gs**4
      abb2(9)=abb2(7)*abb2(8)
      abb2(10)=c3*abb2(2)
      abb2(11)=abb2(10)*abb2(6)
      abb2(12)=spbe2e1*abb2(5)
      abb2(13)=abb2(9)*abb2(11)*abb2(12)
      abb2(14)=1.0_ki/2.0_ki*c1
      abb2(15)=abb2(14)*abb2(6)
      abb2(16)=c2*abb2(6)
      abb2(17)=abb2(15)-abb2(16)
      abb2(18)=abb2(8)*abb2(2)
      abb2(19)=NC*abb2(18)
      abb2(7)=abb2(19)*abb2(7)
      abb2(20)=abb2(7)*abb2(12)
      abb2(21)=-abb2(20)*abb2(17)
      abb2(21)=1.0_ki/2.0_ki*abb2(13)+abb2(21)
      abb2(21)=spae2l3*spae1k2*abb2(21)
      abb2(22)=abb2(6)*abb2(5)
      abb2(23)=spbe2e1*spae1e2
      abb2(24)=abb2(9)*abb2(23)*abb2(10)*abb2(22)
      abb2(25)=c1*abb2(6)
      abb2(26)=-abb2(25)-abb2(16)
      abb2(27)=abb2(23)*abb2(5)
      abb2(26)=abb2(27)*abb2(7)*abb2(26)
      abb2(24)=-2.0_ki*abb2(24)+abb2(26)
      abb2(24)=spak2l3*abb2(24)
      abb2(26)=abb2(16)*abb2(20)
      abb2(13)=-abb2(13)+abb2(26)
      abb2(20)=-abb2(25)*abb2(20)
      abb2(13)=1.0_ki/2.0_ki*abb2(13)+abb2(20)
      abb2(13)=spae1l3*spae2k2*abb2(13)
      abb2(13)=abb2(24)+abb2(13)+abb2(21)
      abb2(13)=spbl3k2*abb2(13)
      abb2(20)=abb2(19)*spae1e2
      abb2(21)=abb2(20)*abb2(14)
      abb2(24)=abb2(8)*spae1e2
      abb2(26)=abb2(10)*abb2(24)
      abb2(28)=abb2(20)*c2
      abb2(21)=-1.0_ki/2.0_ki*abb2(26)+abb2(21)-abb2(28)
      abb2(29)=-spbl3e2*spbl4e1*abb2(21)
      abb2(30)=c1+c2
      abb2(31)=abb2(30)*abb2(23)*abb2(19)
      abb2(32)=abb2(26)*spbe2e1
      abb2(31)=2.0_ki*abb2(32)+abb2(31)
      abb2(32)=spbl4l3*abb2(31)
      abb2(26)=abb2(28)-abb2(26)
      abb2(20)=abb2(20)*c1
      abb2(20)=-abb2(20)+1.0_ki/2.0_ki*abb2(26)
      abb2(26)=spbl3e1*spbl4e2*abb2(20)
      abb2(26)=abb2(32)+abb2(26)+abb2(29)
      abb2(26)=spal3l5*abb2(26)
      abb2(28)=abb2(9)*NC
      abb2(29)=abb2(19)*mT
      abb2(28)=abb2(29)+abb2(28)
      abb2(28)=abb2(28)*abb2(2)
      abb2(29)=abb2(28)*abb2(16)
      abb2(32)=abb2(8)*mT
      abb2(33)=abb2(32)*abb2(2)
      abb2(33)=abb2(33)+abb2(9)
      abb2(11)=abb2(11)*abb2(33)
      abb2(34)=-abb2(28)*abb2(15)
      abb2(34)=abb2(34)+1.0_ki/2.0_ki*abb2(11)+abb2(29)
      abb2(34)=spae1e2*abb2(34)
      abb2(35)=abb2(4)*mH**2*abb2(3)*spak2l5
      abb2(21)=-abb2(21)*abb2(35)
      abb2(21)=abb2(34)+abb2(21)
      abb2(21)=spbk2e2*spbl4e1*abb2(21)
      abb2(34)=abb2(28)*abb2(25)
      abb2(36)=abb2(34)+2.0_ki*abb2(11)+abb2(29)
      abb2(36)=abb2(23)*abb2(36)
      abb2(31)=abb2(31)*abb2(35)
      abb2(31)=abb2(31)+abb2(36)
      abb2(31)=spbl4k2*abb2(31)
      abb2(36)=-abb2(28)*abb2(30)
      abb2(10)=abb2(33)*abb2(10)
      abb2(33)=-2.0_ki*abb2(10)+abb2(36)
      abb2(33)=spak2l5*abb2(27)*abb2(33)
      abb2(11)=-abb2(11)+abb2(29)
      abb2(11)=1.0_ki/2.0_ki*abb2(11)-abb2(34)
      abb2(11)=spae1e2*abb2(11)
      abb2(20)=abb2(20)*abb2(35)
      abb2(11)=abb2(20)+abb2(11)
      abb2(11)=spbk2e1*spbl4e2*abb2(11)
      abb2(20)=abb2(28)*abb2(12)
      abb2(28)=-abb2(14)+c2
      abb2(28)=abb2(20)*abb2(28)
      abb2(10)=abb2(10)*abb2(12)
      abb2(28)=1.0_ki/2.0_ki*abb2(10)+abb2(28)
      abb2(28)=spae2l5*spae1k2*abb2(28)
      abb2(29)=c2*abb2(20)
      abb2(10)=-abb2(10)+abb2(29)
      abb2(20)=-c1*abb2(20)
      abb2(10)=1.0_ki/2.0_ki*abb2(10)+abb2(20)
      abb2(10)=spae1l5*spae2k2*abb2(10)
      abb2(10)=abb2(31)+abb2(10)+abb2(28)+abb2(11)+abb2(33)+abb2(21)+abb2(13)+a&
      &bb2(26)
      abb2(11)=abb2(8)*mT**3
      abb2(13)=abb2(11)*NC
      abb2(7)=abb2(13)+abb2(7)
      abb2(13)=abb2(7)*abb2(25)
      abb2(9)=abb2(9)*abb2(2)
      abb2(9)=abb2(9)+abb2(11)
      abb2(11)=c3*abb2(6)
      abb2(9)=abb2(9)*abb2(11)
      abb2(20)=abb2(7)*abb2(16)
      abb2(21)=abb2(13)+2.0_ki*abb2(9)+abb2(20)
      abb2(21)=abb2(27)*abb2(21)
      abb2(12)=abb2(12)*abb2(24)
      abb2(26)=mT*abb2(12)*c3
      abb2(27)=NC*mT
      abb2(12)=abb2(12)*abb2(27)
      abb2(28)=abb2(12)*abb2(30)
      abb2(28)=2.0_ki*abb2(26)+abb2(28)
      abb2(28)=abb2(28)*abb2(35)
      abb2(21)=abb2(28)+abb2(21)
      abb2(7)=-abb2(7)*abb2(14)*abb2(22)
      abb2(8)=abb2(27)*abb2(8)
      abb2(22)=abb2(5)*abb2(8)
      abb2(28)=abb2(22)*abb2(14)
      abb2(29)=abb2(22)*c2
      abb2(30)=abb2(5)*abb2(32)*c3
      abb2(28)=abb2(29)-abb2(28)+1.0_ki/2.0_ki*abb2(30)
      abb2(31)=abb2(28)*abb2(35)
      abb2(33)=1.0_ki/2.0_ki*abb2(5)
      abb2(33)=abb2(33)*abb2(9)
      abb2(34)=abb2(5)*abb2(20)
      abb2(7)=abb2(31)+abb2(7)+abb2(33)+abb2(34)
      abb2(7)=spbk2e2*abb2(7)
      abb2(28)=spbl3e2*spal3l5*abb2(28)
      abb2(7)=abb2(28)+abb2(7)
      abb2(7)=spae1k2*abb2(7)
      abb2(18)=abb2(18)+abb2(32)
      abb2(18)=abb2(18)*c3
      abb2(19)=abb2(8)+abb2(19)
      abb2(28)=abb2(19)*c2
      abb2(31)=-abb2(18)+abb2(28)
      abb2(33)=-c1*abb2(19)
      abb2(31)=1.0_ki/2.0_ki*abb2(31)+abb2(33)
      abb2(31)=spae1l5*abb2(31)
      abb2(32)=abb2(11)*abb2(32)
      abb2(33)=abb2(8)*abb2(16)
      abb2(33)=-abb2(32)+abb2(33)
      abb2(25)=-abb2(8)*abb2(25)
      abb2(25)=1.0_ki/2.0_ki*abb2(33)+abb2(25)
      abb2(25)=spae1l3*spbl3k2*abb2(25)
      abb2(25)=abb2(25)+abb2(31)
      abb2(25)=spbl4e2*abb2(25)
      abb2(7)=abb2(25)+abb2(7)
      abb2(9)=abb2(9)-abb2(20)
      abb2(9)=1.0_ki/2.0_ki*abb2(9)+abb2(13)
      abb2(9)=abb2(5)*abb2(9)
      abb2(13)=abb2(30)-abb2(29)
      abb2(20)=abb2(22)*c1
      abb2(13)=abb2(20)+1.0_ki/2.0_ki*abb2(13)
      abb2(20)=abb2(13)*abb2(35)
      abb2(9)=abb2(20)+abb2(9)
      abb2(9)=spbk2e1*abb2(9)
      abb2(13)=spbl3e1*spal3l5*abb2(13)
      abb2(9)=abb2(13)+abb2(9)
      abb2(9)=spae2k2*abb2(9)
      abb2(13)=abb2(19)*abb2(14)
      abb2(19)=abb2(13)-1.0_ki/2.0_ki*abb2(18)-abb2(28)
      abb2(19)=spae2l5*abb2(19)
      abb2(8)=abb2(8)*abb2(17)
      abb2(8)=-1.0_ki/2.0_ki*abb2(32)+abb2(8)
      abb2(8)=spae2l3*spbl3k2*abb2(8)
      abb2(8)=abb2(8)+abb2(19)
      abb2(8)=spbl4e1*abb2(8)
      abb2(8)=abb2(8)+abb2(9)
      abb2(9)=-abb2(13)-abb2(18)-1.0_ki/2.0_ki*abb2(28)
      abb2(9)=abb2(23)*abb2(9)
      abb2(13)=-abb2(15)-1.0_ki/2.0_ki*abb2(16)
      abb2(13)=abb2(13)*abb2(27)
      abb2(11)=-mT*abb2(11)
      abb2(11)=abb2(11)+abb2(13)
      abb2(11)=spbl3k2*spbe2e1*abb2(24)*abb2(11)
      abb2(13)=abb2(14)+1.0_ki/2.0_ki*c2
      abb2(12)=abb2(12)*abb2(13)
      abb2(12)=abb2(26)+abb2(12)
      abb2(12)=spal3l5*abb2(12)
      R2d2=0.0_ki
      rat2 = rat2 + R2d2
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='2' value='", &
          & R2d2, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd2h4
