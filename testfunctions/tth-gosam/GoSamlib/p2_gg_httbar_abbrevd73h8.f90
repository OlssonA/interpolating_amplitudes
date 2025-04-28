module     p2_gg_httbar_abbrevd73h8
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_kinematics, only: epstensor
   use p2_gg_httbar_globalsh8
   implicit none
   private
   complex(ki), dimension(53), public :: abb73
   complex(ki), public :: R2d73
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
      abb73(1)=1.0_ki/(-mT**2+es51)
      abb73(2)=sqrt(mT**2)
      abb73(3)=NC**(-1)
      abb73(4)=spak2l5**(-1)
      abb73(5)=spbl5k2**(-1)
      abb73(6)=spak2l3**(-1)
      abb73(7)=spbl3k2**(-1)
      abb73(8)=spbl4k2**(-1)
      abb73(9)=abb73(2)**2
      abb73(10)=abb73(4)*spae1k2
      abb73(11)=abb73(10)*mT
      abb73(12)=abb73(9)*abb73(11)
      abb73(13)=abb73(12)*spae2l5
      abb73(14)=abb73(2)**3
      abb73(15)=spae1e2*abb73(14)
      abb73(13)=abb73(15)-abb73(13)
      abb73(15)=c2*e*gHT*abb73(3)*gs**4*i_*TR
      abb73(16)=abb73(15)*abb73(1)
      abb73(17)=spbl5e1*abb73(16)
      abb73(13)=abb73(17)*abb73(13)
      abb73(15)=abb73(1)*abb73(15)*mT
      abb73(18)=abb73(10)*abb73(15)
      abb73(19)=abb73(18)*spbk1e1
      abb73(20)=abb73(19)*abb73(9)
      abb73(21)=abb73(20)*spak1e2
      abb73(13)=-abb73(21)+abb73(13)
      abb73(21)=-spal3l4*abb73(13)
      abb73(22)=abb73(17)*spae2l4
      abb73(23)=abb73(14)*abb73(22)
      abb73(24)=spae1l3*abb73(23)
      abb73(25)=abb73(12)*abb73(22)
      abb73(26)=-spal3l5*abb73(25)
      abb73(27)=abb73(19)*spae2l4
      abb73(28)=abb73(27)*abb73(9)
      abb73(29)=-spak1l3*abb73(28)
      abb73(21)=abb73(29)+abb73(21)+abb73(26)+abb73(24)
      abb73(21)=spbl3e2*abb73(21)
      abb73(24)=spak2l4*spbk2e2
      abb73(26)=abb73(17)*abb73(2)
      abb73(29)=abb73(24)*abb73(26)
      abb73(30)=abb73(17)*abb73(8)
      abb73(31)=abb73(30)*mT
      abb73(32)=abb73(31)*spbk2e2
      abb73(33)=abb73(32)*abb73(9)
      abb73(29)=abb73(29)+abb73(33)
      abb73(34)=spbl5l3*spae1l5
      abb73(35)=spbl3k1*spae1k1
      abb73(34)=abb73(34)+abb73(35)
      abb73(35)=-abb73(29)*abb73(34)
      abb73(36)=abb73(18)*abb73(9)
      abb73(37)=abb73(24)*abb73(36)
      abb73(10)=abb73(10)*mT**2
      abb73(38)=abb73(16)*abb73(10)
      abb73(39)=abb73(38)*abb73(14)
      abb73(40)=abb73(8)*spbk2e2
      abb73(41)=abb73(40)*abb73(39)
      abb73(42)=-abb73(41)-abb73(37)
      abb73(42)=spbl3e1*abb73(42)
      abb73(35)=abb73(42)+abb73(35)
      abb73(35)=spae2l3*abb73(35)
      abb73(30)=abb73(14)*abb73(10)*abb73(30)*spbk2e2
      abb73(42)=mH**2*abb73(7)*abb73(6)
      abb73(43)=abb73(42)+1.0_ki
      abb73(43)=abb73(43)*spak2l4
      abb73(44)=abb73(17)*spbk2e2
      abb73(12)=abb73(12)*abb73(44)*abb73(43)
      abb73(12)=abb73(30)+abb73(12)
      abb73(12)=spae2l5*abb73(12)
      abb73(30)=abb73(31)*abb73(9)
      abb73(45)=spbl3k2*spae2l3
      abb73(46)=abb73(45)*abb73(30)
      abb73(46)=abb73(46)+abb73(23)
      abb73(47)=spbl5e2*spae1l5
      abb73(48)=spbe2k1*spae1k1
      abb73(47)=abb73(47)+abb73(48)
      abb73(46)=-abb73(46)*abb73(47)
      abb73(48)=abb73(14)*abb73(16)
      abb73(9)=abb73(15)*abb73(9)
      abb73(9)=abb73(48)-abb73(9)
      abb73(48)=abb73(42)*spbk2e2
      abb73(49)=spbl5e1*spae2l4*spae1k2
      abb73(50)=abb73(49)*abb73(48)
      abb73(51)=abb73(49)*spbk2e2
      abb73(50)=abb73(51)+abb73(50)
      abb73(9)=abb73(9)*abb73(50)
      abb73(50)=abb73(2)**4
      abb73(52)=-abb73(50)*abb73(32)
      abb73(14)=abb73(14)*abb73(44)
      abb73(43)=-abb73(14)*abb73(43)
      abb73(43)=abb73(52)+abb73(43)
      abb73(43)=spae1e2*abb73(43)
      abb73(52)=spbk1e1*abb73(41)
      abb73(48)=abb73(48)+spbk2e2
      abb73(20)=spak2l4*abb73(20)*abb73(48)
      abb73(20)=abb73(52)+abb73(20)
      abb73(20)=spak1e2*abb73(20)
      abb73(52)=abb73(10)*abb73(26)
      abb73(53)=abb73(24)*abb73(52)*abb73(45)
      abb73(14)=-spae2l4*abb73(10)*abb73(14)
      abb73(14)=abb73(14)+abb73(53)
      abb73(14)=abb73(5)*abb73(14)
      abb73(18)=-abb73(50)*abb73(18)*spae2l4
      abb73(39)=abb73(8)*abb73(39)
      abb73(50)=-abb73(45)*abb73(39)
      abb73(18)=abb73(18)+abb73(50)
      abb73(18)=spbe2e1*abb73(18)
      abb73(48)=-spak1k2*abb73(28)*abb73(48)
      abb73(9)=abb73(48)+abb73(18)+abb73(14)+abb73(20)+abb73(12)+abb73(43)+abb7&
      &3(21)+abb73(35)+abb73(9)+abb73(46)
      abb73(12)=-spae1e2*abb73(29)
      abb73(14)=abb73(52)*abb73(40)
      abb73(18)=abb73(11)*abb73(44)*spak2l4
      abb73(18)=abb73(14)+abb73(18)
      abb73(18)=spae2l5*abb73(18)
      abb73(20)=abb73(38)*abb73(2)
      abb73(21)=abb73(20)*abb73(40)
      abb73(29)=abb73(21)*spbk1e1
      abb73(35)=abb73(19)*abb73(24)
      abb73(35)=abb73(29)+abb73(35)
      abb73(35)=spak1e2*abb73(35)
      abb73(10)=abb73(10)*abb73(5)
      abb73(38)=-spbk2e2*abb73(10)
      abb73(38)=abb73(38)-abb73(47)
      abb73(40)=abb73(26)*spae2l4
      abb73(38)=abb73(40)*abb73(38)
      abb73(16)=abb73(16)*abb73(2)
      abb73(15)=abb73(16)-abb73(15)
      abb73(16)=abb73(15)*abb73(51)
      abb73(43)=-spbe2e1*abb73(36)*spae2l4
      abb73(44)=abb73(27)*spak1k2
      abb73(46)=-spbk2e2*abb73(44)
      abb73(12)=abb73(46)+abb73(43)+abb73(35)+abb73(18)+abb73(16)+abb73(12)+abb&
      &73(38)
      abb73(16)=-3.0_ki*abb73(41)-2.0_ki*abb73(37)
      abb73(18)=3.0_ki*abb73(23)
      abb73(23)=2.0_ki*abb73(26)
      abb73(35)=abb73(23)*abb73(24)
      abb73(35)=abb73(35)+3.0_ki*abb73(33)
      abb73(37)=spae1l5*abb73(35)
      abb73(38)=abb73(32)*spae1l5
      abb73(25)=3.0_ki*abb73(25)
      abb73(22)=abb73(22)*abb73(11)
      abb73(41)=abb73(36)*spbl3e1
      abb73(43)=abb73(26)*abb73(34)
      abb73(43)=abb73(41)+abb73(43)
      abb73(43)=spae2l3*abb73(43)
      abb73(46)=spbl5k2*spae1l5
      abb73(48)=spbk2k1*spae1k1
      abb73(46)=abb73(46)+abb73(48)
      abb73(48)=spae2k2*abb73(42)*abb73(26)*abb73(46)
      abb73(50)=abb73(52)*abb73(5)
      abb73(45)=-abb73(45)*abb73(50)
      abb73(13)=abb73(48)+3.0_ki*abb73(13)+abb73(45)+abb73(43)
      abb73(19)=abb73(19)*spak1e2
      abb73(11)=spae2l5*abb73(11)*abb73(17)
      abb73(17)=abb73(26)*spae1e2
      abb73(11)=abb73(17)-abb73(19)-abb73(11)
      abb73(17)=2.0_ki*abb73(36)
      abb73(19)=-spae1l5*abb73(23)
      abb73(26)=-spae1l3*abb73(33)
      abb73(14)=spal3l5*abb73(14)
      abb73(29)=spak1l3*abb73(29)
      abb73(33)=spbl5k2*abb73(38)
      abb73(32)=abb73(32)*spae1k1
      abb73(36)=spbk2k1*abb73(32)
      abb73(33)=abb73(33)+abb73(36)
      abb73(33)=spak2l3*abb73(33)
      abb73(14)=abb73(33)+abb73(29)+abb73(26)+abb73(14)
      abb73(26)=spak2l4*abb73(11)
      abb73(15)=-abb73(15)*abb73(49)
      abb73(10)=abb73(40)*abb73(10)
      abb73(10)=abb73(44)+abb73(10)+abb73(15)+abb73(26)
      abb73(10)=spbl3k2*abb73(10)
      abb73(15)=-abb73(40)*abb73(34)
      abb73(26)=-spae2l4*abb73(41)
      abb73(10)=abb73(26)+abb73(10)+abb73(15)
      abb73(15)=abb73(52)*abb73(8)
      abb73(26)=-spal3l5*abb73(15)
      abb73(20)=abb73(20)*abb73(8)
      abb73(29)=abb73(20)*spbk1e1
      abb73(33)=-spak1l3*abb73(29)
      abb73(26)=abb73(33)+abb73(26)
      abb73(26)=spbl3e2*abb73(26)
      abb73(33)=spae1l3*spbl3e2
      abb73(33)=abb73(33)-3.0_ki*abb73(47)
      abb73(33)=abb73(30)*abb73(33)
      abb73(34)=2.0_ki*abb73(50)
      abb73(24)=abb73(24)*abb73(34)
      abb73(36)=spbe2e1*abb73(39)
      abb73(24)=abb73(24)-3.0_ki*abb73(36)+abb73(33)+abb73(26)
      abb73(26)=abb73(31)*abb73(47)
      abb73(20)=abb73(20)*spbe2e1
      abb73(20)=abb73(20)+abb73(26)
      abb73(26)=-2.0_ki*abb73(42)*abb73(20)
      abb73(30)=2.0_ki*abb73(30)
      abb73(15)=2.0_ki*abb73(15)
      abb73(31)=abb73(40)*abb73(42)
      abb73(33)=-abb73(31)*abb73(46)
      abb73(36)=-abb73(42)*abb73(21)
      abb73(35)=-spae1k1*abb73(35)
      abb73(23)=spae1k1*abb73(23)
      abb73(28)=-3.0_ki*abb73(28)
      abb73(29)=-2.0_ki*abb73(29)
      abb73(39)=abb73(42)*abb73(22)
      abb73(41)=abb73(42)*abb73(11)
      abb73(43)=abb73(42)*abb73(38)
      abb73(44)=-abb73(42)*abb73(32)
      abb73(42)=-abb73(42)*abb73(27)
      R2d73=0.0_ki
      rat2 = rat2 + R2d73
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='73' value='", &
          & R2d73, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd73h8
