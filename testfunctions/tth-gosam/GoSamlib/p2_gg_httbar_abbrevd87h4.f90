module     p2_gg_httbar_abbrevd87h4
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_kinematics, only: epstensor
   use p2_gg_httbar_globalsh4
   implicit none
   private
   complex(ki), dimension(55), public :: abb87
   complex(ki), public :: R2d87
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
      abb87(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb87(2)=sqrt(mT**2)
      abb87(3)=NC**(-1)
      abb87(4)=spak2l4**(-1)
      abb87(5)=spbl5k2**(-1)
      abb87(6)=spak2l3**(-1)
      abb87(7)=spbl3k2**(-1)
      abb87(8)=abb87(2)**3
      abb87(9)=i_*TR*e*gHT*abb87(1)*gs**4
      abb87(10)=abb87(8)*abb87(9)
      abb87(11)=abb87(9)*mT
      abb87(12)=abb87(2)**2
      abb87(13)=abb87(11)*abb87(12)
      abb87(14)=abb87(10)+abb87(13)
      abb87(15)=c1*abb87(3)
      abb87(15)=abb87(15)-c3
      abb87(14)=-abb87(14)*abb87(15)
      abb87(16)=mT**2
      abb87(17)=abb87(16)*abb87(5)
      abb87(18)=-abb87(14)*abb87(17)
      abb87(19)=spae2k2*abb87(4)
      abb87(20)=abb87(19)*spbe2e1
      abb87(21)=-abb87(20)*abb87(18)
      abb87(12)=abb87(12)*abb87(9)
      abb87(22)=abb87(9)*abb87(2)
      abb87(23)=abb87(22)*mT
      abb87(12)=abb87(23)+abb87(12)
      abb87(23)=-mT*abb87(15)
      abb87(12)=-abb87(12)*abb87(23)
      abb87(24)=abb87(12)*abb87(20)
      abb87(25)=abb87(24)*spak2l5
      abb87(21)=abb87(21)+abb87(25)
      abb87(21)=spae1k1*abb87(21)
      abb87(26)=spbl3k2*abb87(5)
      abb87(16)=abb87(26)*abb87(16)
      abb87(27)=-abb87(22)*abb87(15)
      abb87(28)=-abb87(27)*abb87(16)
      abb87(29)=abb87(28)*spae1k1
      abb87(30)=abb87(29)*abb87(20)
      abb87(31)=spak2l3*abb87(30)
      abb87(13)=-abb87(13)*abb87(15)
      abb87(32)=spak2l5*mH**2*abb87(7)*abb87(6)
      abb87(33)=abb87(13)*abb87(32)
      abb87(34)=abb87(20)*spae1k1
      abb87(35)=abb87(34)*abb87(33)
      abb87(21)=abb87(31)+abb87(21)+abb87(35)
      abb87(21)=spbk2k1*abb87(21)
      abb87(31)=spak1l5*spbk1e1
      abb87(35)=-abb87(14)*abb87(31)
      abb87(10)=abb87(15)*abb87(10)
      abb87(36)=abb87(10)*spal3l5
      abb87(37)=spbl3e1*abb87(36)
      abb87(38)=abb87(13)*abb87(26)
      abb87(39)=spak1l3*spbk1e1
      abb87(40)=-abb87(38)*abb87(39)
      abb87(35)=abb87(40)+abb87(37)+abb87(35)
      abb87(37)=spbl4e2*spae1e2
      abb87(35)=abb87(37)*abb87(35)
      abb87(40)=abb87(12)*abb87(5)
      abb87(41)=abb87(32)*abb87(27)
      abb87(40)=abb87(40)-abb87(41)
      abb87(41)=-spbk1e1*abb87(40)
      abb87(42)=spbk2e2*spae1e2
      abb87(43)=abb87(42)*spbl4k2
      abb87(44)=abb87(43)*abb87(41)
      abb87(27)=abb87(27)*spal3l5
      abb87(45)=abb87(27)*spbk1e1
      abb87(46)=abb87(45)*abb87(37)
      abb87(47)=-spbl3k2*abb87(46)
      abb87(48)=abb87(27)*spbl3e2
      abb87(49)=abb87(48)*spae1e2
      abb87(50)=abb87(49)*spbl4k2
      abb87(51)=spbk1e1*abb87(50)
      abb87(44)=abb87(51)+abb87(47)+abb87(44)
      abb87(44)=spak1k2*abb87(44)
      abb87(16)=abb87(10)*abb87(16)
      abb87(47)=-spae2l3*abb87(16)
      abb87(8)=abb87(8)*abb87(11)
      abb87(9)=abb87(9)*abb87(2)**4
      abb87(8)=abb87(9)+abb87(8)
      abb87(8)=-abb87(8)*abb87(23)
      abb87(9)=-spae2l5*abb87(8)
      abb87(9)=abb87(47)+abb87(9)
      abb87(23)=spbe2e1*abb87(4)
      abb87(47)=abb87(23)*spae1k2
      abb87(9)=abb87(47)*abb87(9)
      abb87(51)=abb87(8)*abb87(5)
      abb87(10)=abb87(10)*abb87(32)
      abb87(10)=abb87(51)+abb87(10)
      abb87(51)=abb87(37)*spbk2e1
      abb87(52)=abb87(10)*abb87(51)
      abb87(53)=spbl4k2*spae2k2
      abb87(54)=abb87(53)*spbe2e1
      abb87(55)=-abb87(14)*abb87(54)
      abb87(8)=abb87(8)*abb87(20)
      abb87(8)=abb87(8)+abb87(55)
      abb87(8)=spae1l5*abb87(8)
      abb87(10)=-abb87(42)*abb87(10)
      abb87(36)=-spbl3e2*spae1e2*abb87(36)
      abb87(10)=abb87(36)+abb87(10)
      abb87(10)=spbl4e1*abb87(10)
      abb87(16)=abb87(20)*abb87(16)
      abb87(36)=-abb87(38)*abb87(54)
      abb87(16)=abb87(16)+abb87(36)
      abb87(16)=spae1l3*abb87(16)
      abb87(13)=abb87(13)*spal3l5
      abb87(36)=spbl3k1*abb87(34)*abb87(13)
      abb87(8)=abb87(36)+abb87(44)+abb87(16)+abb87(10)+abb87(8)+abb87(52)+abb87&
      &(35)+abb87(9)+abb87(21)
      abb87(9)=abb87(28)*spae2l3
      abb87(10)=abb87(12)*spae2l5
      abb87(9)=abb87(10)+abb87(9)
      abb87(16)=-abb87(47)*abb87(9)
      abb87(21)=abb87(40)*abb87(51)
      abb87(35)=-abb87(42)*abb87(40)
      abb87(35)=abb87(49)+abb87(35)
      abb87(35)=spbl4e1*abb87(35)
      abb87(36)=abb87(28)*abb87(20)
      abb87(44)=spae1l3*abb87(36)
      abb87(24)=spae1l5*abb87(24)
      abb87(47)=abb87(27)*abb87(37)
      abb87(51)=-spbl3e1*abb87(47)
      abb87(16)=abb87(51)+abb87(44)+abb87(35)+abb87(24)+abb87(21)+abb87(16)
      abb87(21)=abb87(14)*spae1l5
      abb87(24)=abb87(38)*spae1l3
      abb87(21)=abb87(24)+abb87(21)
      abb87(24)=spbl4e2*abb87(21)
      abb87(35)=abb87(40)*spbk2e2
      abb87(35)=abb87(35)-abb87(48)
      abb87(44)=spbl4k2*abb87(35)
      abb87(27)=abb87(27)*spbl4e2
      abb87(51)=spbl3k2*abb87(27)
      abb87(44)=abb87(51)+abb87(44)
      abb87(44)=spae1k2*abb87(44)
      abb87(24)=abb87(44)+abb87(24)
      abb87(44)=-spak2l3*spbk2e1*abb87(28)
      abb87(51)=abb87(13)*spbl3e1
      abb87(44)=abb87(44)-abb87(51)
      abb87(44)=abb87(19)*abb87(44)
      abb87(22)=abb87(11)+abb87(22)
      abb87(22)=-abb87(22)*abb87(15)
      abb87(31)=abb87(22)*abb87(31)
      abb87(11)=-abb87(11)*abb87(15)
      abb87(15)=abb87(11)*abb87(26)
      abb87(26)=abb87(39)*abb87(15)
      abb87(26)=abb87(31)+abb87(26)
      abb87(31)=-abb87(53)*abb87(26)
      abb87(18)=abb87(18)-abb87(33)
      abb87(33)=abb87(19)*abb87(18)
      abb87(39)=abb87(12)*abb87(19)
      abb87(52)=-spak2l5*abb87(39)
      abb87(33)=abb87(52)+abb87(33)
      abb87(33)=spbk2e1*abb87(33)
      abb87(31)=abb87(33)+abb87(31)+abb87(44)
      abb87(14)=-abb87(14)*abb87(37)
      abb87(33)=spbe2k1*spae1k1*abb87(39)
      abb87(14)=abb87(14)+abb87(33)
      abb87(33)=-abb87(22)*abb87(53)
      abb87(33)=2.0_ki*abb87(39)+abb87(33)
      abb87(21)=spbe2e1*abb87(21)
      abb87(39)=spbk2e2*abb87(41)
      abb87(44)=spbk1e1*abb87(48)
      abb87(39)=abb87(44)+abb87(39)
      abb87(39)=spak1e2*abb87(39)
      abb87(35)=-2.0_ki*abb87(35)
      abb87(13)=abb87(20)*abb87(13)
      abb87(44)=spak1e2*spbl4e2
      abb87(45)=-abb87(45)*abb87(44)
      abb87(13)=abb87(13)+abb87(45)
      abb87(27)=-2.0_ki*abb87(27)
      abb87(45)=spbe2k1*abb87(19)*abb87(29)
      abb87(38)=-abb87(37)*abb87(38)
      abb87(38)=abb87(38)+abb87(45)
      abb87(19)=abb87(19)*abb87(28)
      abb87(28)=-abb87(53)*abb87(15)
      abb87(19)=2.0_ki*abb87(19)+abb87(28)
      abb87(28)=spbk2e1*abb87(18)
      abb87(28)=abb87(28)-abb87(51)
      abb87(28)=abb87(28)*spae1e2*abb87(4)
      abb87(17)=abb87(22)*abb87(17)
      abb87(45)=-abb87(11)*abb87(32)
      abb87(45)=-abb87(17)+abb87(45)
      abb87(45)=spbk2k1*abb87(45)
      abb87(48)=-spbl3k1*spal3l5*abb87(11)
      abb87(45)=abb87(45)+abb87(48)
      abb87(48)=spae1k1*abb87(4)
      abb87(45)=abb87(48)*abb87(45)
      abb87(11)=abb87(11)*abb87(4)
      abb87(51)=-spal3l5*abb87(11)
      abb87(18)=-abb87(20)*abb87(18)
      abb87(20)=-abb87(41)*abb87(44)
      abb87(36)=spak2l3*abb87(36)
      abb87(18)=abb87(36)+abb87(20)+abb87(25)+abb87(18)
      abb87(20)=2.0_ki*spbl4e2
      abb87(20)=abb87(40)*abb87(20)
      abb87(17)=-abb87(4)*abb87(17)
      abb87(11)=-abb87(32)*abb87(11)
      abb87(11)=abb87(17)+abb87(11)
      abb87(17)=abb87(29)*spae2l3
      abb87(25)=-abb87(4)*abb87(17)
      abb87(29)=-abb87(48)*abb87(10)
      abb87(25)=abb87(25)+abb87(29)
      abb87(25)=spbe2k1*abb87(25)
      abb87(29)=-abb87(43)*abb87(40)
      abb87(32)=-spbl3k2*abb87(47)
      abb87(25)=abb87(50)+abb87(32)+abb87(29)+abb87(25)
      abb87(9)=-2.0_ki*abb87(4)*abb87(9)
      abb87(12)=-abb87(12)*abb87(34)
      abb87(10)=spae1k1*abb87(10)
      abb87(10)=abb87(10)+abb87(17)
      abb87(10)=abb87(23)*abb87(10)
      abb87(17)=-abb87(42)*abb87(41)
      abb87(23)=abb87(49)*spbk1e1
      abb87(17)=-abb87(23)+abb87(17)
      abb87(23)=abb87(37)*abb87(41)
      R2d87=0.0_ki
      rat2 = rat2 + R2d87
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='87' value='", &
          & R2d87, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd87h4
