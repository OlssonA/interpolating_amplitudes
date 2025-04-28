module     p2_gg_httbar_abbrevd257h12
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_kinematics, only: epstensor
   use p2_gg_httbar_globalsh12
   implicit none
   private
   complex(ki), dimension(58), public :: abb257
   complex(ki), public :: R2d257
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
      abb257(1)=sqrt(mT**2)
      abb257(2)=NC**(-1)
      abb257(3)=spak2l5**(-1)
      abb257(4)=spak2l4**(-1)
      abb257(5)=spak2l3**(-1)
      abb257(6)=spbl3k2**(-1)
      abb257(7)=spbl5e1*spae1e2
      abb257(8)=c2*abb257(2)
      abb257(8)=abb257(8)-c3
      abb257(8)=abb257(8)*gs**4*i_*TR*e*gHT
      abb257(9)=-abb257(8)*abb257(1)**2
      abb257(10)=abb257(7)*abb257(9)
      abb257(11)=abb257(10)*spbl4k1
      abb257(12)=spae1e2*spbl4e1
      abb257(13)=abb257(12)*abb257(9)
      abb257(14)=abb257(13)*spbl5k1
      abb257(11)=abb257(11)-abb257(14)
      abb257(14)=abb257(5)*abb257(6)*mH**2
      abb257(15)=abb257(14)*spak1k2
      abb257(16)=-abb257(15)+2.0_ki*spak1k2
      abb257(16)=-spbk2e2*abb257(16)*abb257(11)
      abb257(17)=abb257(14)-2.0_ki
      abb257(18)=spbl5k2*abb257(17)*spae2k2
      abb257(19)=spae2l3*spbl5l3
      abb257(18)=abb257(18)+abb257(19)
      abb257(20)=-abb257(8)*abb257(1)**3
      abb257(21)=abb257(4)*mT
      abb257(22)=abb257(21)*spae1k2
      abb257(23)=abb257(20)*abb257(22)
      abb257(24)=-spbe2e1*abb257(23)*abb257(18)
      abb257(25)=-abb257(1)*abb257(8)
      abb257(26)=abb257(25)*spak2l3
      abb257(27)=abb257(3)*mT
      abb257(28)=abb257(27)*spae2k2
      abb257(29)=abb257(26)*abb257(28)
      abb257(30)=abb257(29)*spbl3k2
      abb257(31)=abb257(20)*abb257(28)
      abb257(30)=abb257(30)-abb257(31)
      abb257(30)=abb257(30)*spbe2e1
      abb257(32)=spae1k1*spbl4k1
      abb257(33)=-abb257(30)*abb257(32)
      abb257(34)=spak1l3*spbl3e2*abb257(11)
      abb257(35)=spbl5e2*abb257(12)
      abb257(7)=-spbl4e2*abb257(7)
      abb257(7)=abb257(7)+abb257(35)
      abb257(7)=abb257(7)*abb257(8)*abb257(1)**4
      abb257(12)=abb257(12)*abb257(27)
      abb257(35)=spbl3e2*spak2l3*abb257(12)
      abb257(36)=abb257(21)*spae1e2
      abb257(37)=abb257(36)*spbl5e2
      abb257(38)=abb257(37)*spbk1e1*spak1k2
      abb257(35)=abb257(35)+abb257(38)
      abb257(35)=abb257(20)*abb257(35)
      abb257(36)=abb257(25)*abb257(36)
      abb257(38)=spbk1e1*abb257(36)*spak1k2
      abb257(39)=spbl5l3*spbk2e2
      abb257(40)=abb257(39)*spak2l3
      abb257(41)=abb257(38)*abb257(40)
      abb257(42)=abb257(9)*spbl5e1
      abb257(43)=abb257(42)*spbl4e2
      abb257(9)=abb257(9)*spbl4e1
      abb257(44)=abb257(9)*spbl5e2
      abb257(43)=abb257(43)-abb257(44)
      abb257(44)=abb257(43)*spae2k2
      abb257(45)=spae1l3*spbl3k2*abb257(44)
      abb257(7)=abb257(34)+abb257(45)+abb257(33)+abb257(41)+abb257(35)+abb257(2&
      &4)+abb257(16)+abb257(7)
      abb257(16)=abb257(25)*abb257(28)
      abb257(24)=abb257(16)*spbe2e1
      abb257(28)=abb257(24)*abb257(32)
      abb257(33)=2.0_ki*abb257(28)
      abb257(34)=-3.0_ki*abb257(13)+abb257(38)
      abb257(34)=spbl5e2*abb257(34)
      abb257(35)=spbl4e2*abb257(10)
      abb257(28)=abb257(28)+3.0_ki*abb257(35)+abb257(34)
      abb257(34)=-abb257(22)*abb257(26)*abb257(39)
      abb257(35)=-spbl5e2*abb257(23)
      abb257(26)=abb257(26)*abb257(27)
      abb257(39)=abb257(32)*abb257(26)
      abb257(41)=spbl3e2*abb257(39)
      abb257(34)=abb257(41)+abb257(35)+abb257(34)
      abb257(22)=abb257(22)*abb257(25)
      abb257(35)=-spbl5e2*abb257(22)
      abb257(41)=abb257(25)*abb257(21)
      abb257(45)=spak1k2*abb257(41)
      abb257(18)=-spbk1e1*abb257(45)*abb257(18)
      abb257(29)=spbl4e1*abb257(29)
      abb257(46)=abb257(8)*spbl5e1
      abb257(47)=abb257(46)*spbl4k1
      abb257(8)=abb257(8)*spbl4e1
      abb257(48)=abb257(8)*spbl5k1
      abb257(47)=abb257(47)-abb257(48)
      abb257(48)=-spae2k2*abb257(47)
      abb257(49)=spak1l3*abb257(48)
      abb257(29)=abb257(49)+abb257(29)
      abb257(29)=spbl3k2*abb257(29)
      abb257(31)=-spbl4e1*abb257(31)
      abb257(49)=spbl4k1*abb257(42)
      abb257(50)=-spbl5k1*abb257(9)
      abb257(49)=abb257(49)+abb257(50)
      abb257(49)=spak1e2*abb257(49)
      abb257(18)=abb257(49)+abb257(31)+abb257(29)+abb257(18)
      abb257(16)=abb257(16)*spbl4e1
      abb257(29)=-2.0_ki*abb257(16)
      abb257(31)=-spak1e2*abb257(47)
      abb257(49)=abb257(8)*spae2k2
      abb257(50)=spbl5k2*abb257(49)
      abb257(51)=abb257(46)*spae2k2
      abb257(52)=-spbl4k2*abb257(51)
      abb257(16)=abb257(52)+abb257(31)-abb257(16)+abb257(50)
      abb257(31)=abb257(17)*spbk2e2
      abb257(50)=abb257(9)*spae1k2
      abb257(52)=-abb257(50)*abb257(31)
      abb257(23)=spbe2e1*abb257(23)
      abb257(53)=spae1l3*spbl3e2
      abb257(54)=-abb257(9)*abb257(53)
      abb257(23)=abb257(54)+abb257(52)-2.0_ki*abb257(23)
      abb257(45)=spbk1e1*abb257(45)
      abb257(9)=3.0_ki*abb257(9)-2.0_ki*abb257(45)
      abb257(45)=abb257(42)*spae1k2
      abb257(52)=abb257(45)*abb257(31)
      abb257(53)=abb257(42)*abb257(53)
      abb257(52)=abb257(52)+abb257(53)
      abb257(42)=-3.0_ki*abb257(42)
      abb257(53)=2.0_ki*abb257(24)
      abb257(54)=abb257(26)*spbl3e2
      abb257(39)=spbe2e1*abb257(39)
      abb257(43)=-spae1l3*abb257(43)
      abb257(39)=abb257(39)+abb257(43)
      abb257(43)=-spbl4e1*abb257(26)
      abb257(55)=spak1l3*abb257(47)
      abb257(43)=abb257(43)+abb257(55)
      abb257(26)=abb257(26)*spbe2e1
      abb257(55)=-spbl5l3*abb257(38)
      abb257(56)=spbl5l3*abb257(22)
      abb257(50)=spbl5e2*abb257(50)
      abb257(45)=-spbl4e2*abb257(45)
      abb257(45)=abb257(50)+abb257(45)
      abb257(50)=abb257(14)-1.0_ki
      abb257(45)=abb257(50)*abb257(45)
      abb257(15)=abb257(15)-spak1k2
      abb257(15)=abb257(15)*abb257(47)
      abb257(12)=abb257(20)*abb257(12)
      abb257(47)=abb257(50)*spbl5k2
      abb257(57)=-abb257(47)*abb257(38)
      abb257(12)=-2.0_ki*abb257(12)+abb257(57)
      abb257(57)=abb257(22)*abb257(47)
      abb257(58)=2.0_ki*abb257(25)
      abb257(27)=abb257(58)*abb257(27)
      abb257(32)=-abb257(32)*abb257(27)
      abb257(32)=abb257(57)+abb257(32)
      abb257(20)=abb257(20)*abb257(37)
      abb257(37)=abb257(36)*abb257(40)
      abb257(20)=abb257(20)+abb257(37)
      abb257(37)=spbl5e2*abb257(36)
      abb257(14)=-abb257(25)*abb257(14)
      abb257(14)=abb257(58)+abb257(14)
      abb257(14)=spbl5k2*abb257(14)*abb257(21)*spae2k2
      abb257(19)=-abb257(19)*abb257(41)
      abb257(14)=abb257(14)+abb257(19)
      abb257(19)=-abb257(21)*abb257(58)
      abb257(21)=-spbl5l3*abb257(36)
      abb257(25)=-abb257(36)*abb257(47)
      abb257(11)=-2.0_ki*abb257(11)
      abb257(40)=spbl3e2*abb257(13)
      abb257(41)=-spbl3k2*abb257(49)
      abb257(47)=-spbl3e2*abb257(10)
      abb257(49)=spbl3k2*abb257(51)
      abb257(13)=abb257(13)*abb257(17)
      abb257(13)=abb257(13)+abb257(38)
      abb257(13)=spbk2e2*abb257(13)
      abb257(17)=-spbk2e2*abb257(22)
      abb257(22)=abb257(8)*abb257(50)
      abb257(36)=spbk2e2*abb257(36)
      abb257(10)=-abb257(10)*abb257(31)
      abb257(31)=-abb257(46)*abb257(50)
      R2d257=0.0_ki
      rat2 = rat2 + R2d257
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='257' value='", &
          & R2d257, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd257h12
