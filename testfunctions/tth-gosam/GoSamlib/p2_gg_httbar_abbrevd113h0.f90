module     p2_gg_httbar_abbrevd113h0
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_kinematics, only: epstensor
   use p2_gg_httbar_globalsh0
   implicit none
   private
   complex(ki), dimension(54), public :: abb113
   complex(ki), public :: R2d113
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
      abb113(1)=sqrt(mT**2)
      abb113(2)=es45**(-1)
      abb113(3)=spbl4k2**(-1)
      abb113(4)=spbl5k2**(-1)
      abb113(5)=spak2l3**(-1)
      abb113(6)=spbl3k2**(-1)
      abb113(7)=c1-c2
      abb113(8)=gs**4*i_*TR*mT*e*gHT*abb113(2)
      abb113(9)=abb113(7)*abb113(8)*abb113(1)
      abb113(10)=-abb113(4)*abb113(9)
      abb113(11)=abb113(10)*spbl3k2
      abb113(12)=abb113(11)*spal3l4
      abb113(9)=-abb113(3)*abb113(9)
      abb113(13)=abb113(9)*spbl3k2
      abb113(14)=abb113(13)*spal3l5
      abb113(12)=abb113(14)+abb113(12)
      abb113(12)=spae2k2*abb113(12)
      abb113(7)=abb113(8)*abb113(7)*abb113(1)**3
      abb113(8)=-abb113(3)*abb113(7)
      abb113(14)=abb113(8)*spae2l5
      abb113(7)=-abb113(4)*abb113(7)
      abb113(15)=abb113(7)*spae2l4
      abb113(14)=abb113(14)+abb113(15)+abb113(12)
      abb113(15)=spbk2k1*spae1k1
      abb113(16)=abb113(15)*spbe2e1
      abb113(17)=-abb113(16)*abb113(14)
      abb113(18)=abb113(8)*spbk2e1
      abb113(19)=spak2l5*abb113(18)
      abb113(20)=abb113(7)*spbk2e1
      abb113(21)=spak2l4*abb113(20)
      abb113(19)=abb113(19)+abb113(21)
      abb113(21)=abb113(5)*abb113(6)*mH**2
      abb113(22)=abb113(21)-2.0_ki
      abb113(23)=spae1e2*spbk2e2
      abb113(19)=abb113(19)*abb113(22)*abb113(23)
      abb113(24)=abb113(18)*spae1e2
      abb113(25)=spal3l5*abb113(24)
      abb113(26)=abb113(20)*spae1e2
      abb113(27)=spal3l4*abb113(26)
      abb113(25)=abb113(25)+abb113(27)
      abb113(25)=spbl3e2*abb113(25)
      abb113(27)=abb113(8)*spae1l5
      abb113(28)=abb113(7)*spae1l4
      abb113(27)=abb113(27)+abb113(28)
      abb113(28)=spbe2e1*abb113(27)
      abb113(29)=-spae2l3*spbl3k2*abb113(28)
      abb113(30)=abb113(23)*spbk1e1
      abb113(31)=abb113(30)*spak1l5
      abb113(32)=abb113(13)*abb113(31)
      abb113(30)=abb113(30)*spak1l4
      abb113(33)=abb113(11)*abb113(30)
      abb113(32)=abb113(32)+abb113(33)
      abb113(32)=spak2l3*abb113(32)
      abb113(33)=-abb113(8)*abb113(31)
      abb113(34)=-abb113(7)*abb113(30)
      abb113(17)=abb113(32)+abb113(29)+abb113(25)+abb113(34)+abb113(33)+abb113(&
      &17)+abb113(19)
      abb113(19)=abb113(30)*abb113(10)
      abb113(25)=abb113(31)*abb113(9)
      abb113(19)=abb113(19)+abb113(25)
      abb113(25)=2.0_ki*abb113(19)
      abb113(29)=abb113(9)*abb113(16)
      abb113(30)=-spae2l5*abb113(29)
      abb113(31)=abb113(10)*abb113(16)
      abb113(32)=-spae2l4*abb113(31)
      abb113(19)=abb113(32)+abb113(30)-abb113(19)
      abb113(30)=abb113(11)*spae1l4
      abb113(32)=abb113(13)*spae1l5
      abb113(30)=abb113(30)+abb113(32)
      abb113(32)=spak2l3*abb113(30)
      abb113(27)=abb113(32)-abb113(27)
      abb113(27)=spbk2e2*abb113(27)
      abb113(32)=abb113(10)*spbk2e2
      abb113(33)=abb113(32)*spak2l4
      abb113(34)=abb113(9)*spbk2e2
      abb113(35)=abb113(34)*spak2l5
      abb113(33)=-abb113(35)-abb113(33)
      abb113(33)=abb113(33)*abb113(22)*abb113(15)
      abb113(35)=abb113(9)*spal3l5
      abb113(36)=abb113(10)*spal3l4
      abb113(35)=abb113(35)+abb113(36)
      abb113(36)=abb113(35)*spbl3e2
      abb113(37)=-abb113(15)*abb113(36)
      abb113(27)=abb113(37)+abb113(33)+abb113(27)
      abb113(33)=abb113(32)*spae1l4
      abb113(37)=abb113(34)*spae1l5
      abb113(33)=abb113(33)+abb113(37)
      abb113(37)=2.0_ki*abb113(33)
      abb113(12)=-spbk2e1*abb113(12)
      abb113(38)=spak1l4*spbk1e1
      abb113(39)=abb113(38)*abb113(11)
      abb113(40)=spak1l5*spbk1e1
      abb113(41)=abb113(40)*abb113(13)
      abb113(39)=abb113(39)+abb113(41)
      abb113(41)=spae2l3*abb113(39)
      abb113(18)=-spae2l5*abb113(18)
      abb113(20)=-spae2l4*abb113(20)
      abb113(12)=abb113(41)+abb113(20)+abb113(18)+abb113(12)
      abb113(18)=abb113(9)*spbk2e1
      abb113(20)=-spae2l5*abb113(18)
      abb113(41)=abb113(10)*spbk2e1
      abb113(42)=-spae2l4*abb113(41)
      abb113(20)=abb113(20)+abb113(42)
      abb113(24)=-2.0_ki*abb113(24)
      abb113(42)=2.0_ki*abb113(9)
      abb113(43)=abb113(15)*abb113(42)
      abb113(44)=-spak2l3*abb113(13)
      abb113(8)=abb113(8)+abb113(44)
      abb113(8)=abb113(23)*abb113(8)
      abb113(44)=-abb113(23)*abb113(42)
      abb113(45)=abb113(9)*abb113(23)
      abb113(46)=-spae2l3*abb113(13)
      abb113(26)=-2.0_ki*abb113(26)
      abb113(47)=2.0_ki*abb113(10)
      abb113(15)=abb113(15)*abb113(47)
      abb113(48)=-spak2l3*abb113(11)
      abb113(7)=abb113(7)+abb113(48)
      abb113(7)=abb113(23)*abb113(7)
      abb113(48)=-abb113(23)*abb113(47)
      abb113(23)=abb113(10)*abb113(23)
      abb113(49)=-spae2l3*abb113(11)
      abb113(16)=-abb113(16)*abb113(35)
      abb113(50)=-spbk2e1*abb113(35)
      abb113(39)=spae1e2*abb113(39)
      abb113(13)=-spae1e2*abb113(13)
      abb113(11)=-spae1e2*abb113(11)
      abb113(51)=abb113(21)-1.0_ki
      abb113(52)=abb113(51)*spak2l5
      abb113(53)=-abb113(29)*abb113(52)
      abb113(51)=abb113(51)*spak2l4
      abb113(54)=-abb113(31)*abb113(51)
      abb113(28)=abb113(53)+abb113(54)-2.0_ki*abb113(28)
      abb113(21)=abb113(21)*spbk2e1
      abb113(53)=-abb113(9)*abb113(21)
      abb113(53)=abb113(18)+abb113(53)
      abb113(53)=spak2l5*abb113(53)
      abb113(21)=-abb113(10)*abb113(21)
      abb113(21)=abb113(41)+abb113(21)
      abb113(21)=spak2l4*abb113(21)
      abb113(40)=abb113(42)*abb113(40)
      abb113(38)=abb113(47)*abb113(38)
      abb113(21)=abb113(38)+abb113(40)+abb113(53)+abb113(21)
      abb113(14)=spbe2e1*abb113(14)
      abb113(9)=abb113(9)*spbe2e1
      abb113(38)=spae2l5*abb113(9)
      abb113(10)=abb113(10)*spbe2e1
      abb113(40)=spae2l4*abb113(10)
      abb113(38)=abb113(38)+abb113(40)
      abb113(34)=abb113(34)*abb113(22)*spak2l5
      abb113(22)=abb113(32)*abb113(22)*spak2l4
      abb113(22)=abb113(36)+abb113(34)+abb113(22)
      abb113(32)=spbe2e1*abb113(35)
      abb113(34)=abb113(9)*abb113(52)
      abb113(35)=abb113(10)*abb113(51)
      abb113(34)=abb113(34)+abb113(35)
      abb113(29)=spae2k2*abb113(29)
      abb113(18)=spae2k2*abb113(18)
      abb113(9)=-spae2k2*abb113(9)
      abb113(31)=spae2k2*abb113(31)
      abb113(35)=spae2k2*abb113(41)
      abb113(10)=-spae2k2*abb113(10)
      R2d113=0.0_ki
      rat2 = rat2 + R2d113
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='113' value='", &
          & R2d113, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd113h0
