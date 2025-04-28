module     p0_gg_gh_d1h0l1d
   ! file: /mt/home/sjones/repos/POWHEG-BOX/ggh-gosam/GoSam_POWHEG/Virtual/p0_g &
   ! &g_gh/helicity0d1h0l1d.f90
   ! generator: buildfortran_d.py
   use p0_gg_gh_config, only: ki
   use p0_gg_gh_util, only: cond, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, private :: iv0
   integer, private :: iv1
   integer, private :: iv2
   integer, private :: iv3
   real(ki), dimension(4), private :: qshift
   public :: derivative
contains
!---#[ function brack_1:
   pure function brack_1(Q, mu2) result(brack)
      use p0_gg_gh_model
      use p0_gg_gh_kinematics
      use p0_gg_gh_color
      use p0_gg_gh_abbrevd1h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(26) :: acd1
      complex(ki) :: brack
      acd1(1)=dotproduct(k2,qshift)
      acd1(2)=dotproduct(qshift,spvak2k3)
      acd1(3)=abb1(23)
      acd1(4)=abb1(14)
      acd1(5)=dotproduct(qshift,qshift)
      acd1(6)=abb1(15)
      acd1(7)=dotproduct(qshift,spvak1k2)
      acd1(8)=abb1(19)
      acd1(9)=abb1(9)
      acd1(10)=dotproduct(qshift,spvak1k3)
      acd1(11)=dotproduct(qshift,spvak2k1)
      acd1(12)=abb1(11)
      acd1(13)=abb1(17)
      acd1(14)=abb1(12)
      acd1(15)=dotproduct(qshift,spvak2l4)
      acd1(16)=abb1(18)
      acd1(17)=dotproduct(qshift,spval4k3)
      acd1(18)=abb1(13)
      acd1(19)=abb1(10)
      acd1(20)=acd1(3)*acd1(1)
      acd1(21)=acd1(8)*acd1(7)
      acd1(20)=-acd1(9)+acd1(21)+acd1(20)
      acd1(20)=acd1(2)*acd1(20)
      acd1(21)=acd1(12)*acd1(10)
      acd1(21)=-acd1(14)+acd1(21)
      acd1(21)=acd1(11)*acd1(21)
      acd1(22)=-acd1(4)*acd1(1)
      acd1(23)=acd1(6)*acd1(5)
      acd1(24)=-acd1(13)*acd1(10)
      acd1(25)=-acd1(16)*acd1(15)
      acd1(26)=-acd1(18)*acd1(17)
      brack=acd1(19)+acd1(20)+acd1(21)+acd1(22)+acd1(23)+acd1(24)+acd1(25)+acd1&
      &(26)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p0_gg_gh_model
      use p0_gg_gh_kinematics
      use p0_gg_gh_color
      use p0_gg_gh_abbrevd1h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(31) :: acd1
      complex(ki) :: brack
      acd1(1)=k2(iv1)
      acd1(2)=dotproduct(qshift,spvak2k3)
      acd1(3)=abb1(23)
      acd1(4)=abb1(14)
      acd1(5)=qshift(iv1)
      acd1(6)=abb1(15)
      acd1(7)=spvak2k3(iv1)
      acd1(8)=dotproduct(k2,qshift)
      acd1(9)=dotproduct(qshift,spvak1k2)
      acd1(10)=abb1(19)
      acd1(11)=abb1(9)
      acd1(12)=spvak1k2(iv1)
      acd1(13)=spvak1k3(iv1)
      acd1(14)=dotproduct(qshift,spvak2k1)
      acd1(15)=abb1(11)
      acd1(16)=abb1(17)
      acd1(17)=spvak2k1(iv1)
      acd1(18)=dotproduct(qshift,spvak1k3)
      acd1(19)=abb1(12)
      acd1(20)=spvak2l4(iv1)
      acd1(21)=abb1(18)
      acd1(22)=spval4k3(iv1)
      acd1(23)=abb1(13)
      acd1(24)=acd1(3)*acd1(1)
      acd1(25)=acd1(12)*acd1(10)
      acd1(24)=acd1(25)+acd1(24)
      acd1(24)=acd1(2)*acd1(24)
      acd1(25)=acd1(8)*acd1(3)
      acd1(26)=acd1(9)*acd1(10)
      acd1(25)=-acd1(11)+acd1(26)+acd1(25)
      acd1(25)=acd1(7)*acd1(25)
      acd1(26)=acd1(14)*acd1(15)
      acd1(26)=-acd1(16)+acd1(26)
      acd1(26)=acd1(13)*acd1(26)
      acd1(27)=acd1(18)*acd1(15)
      acd1(27)=-acd1(19)+acd1(27)
      acd1(27)=acd1(17)*acd1(27)
      acd1(28)=-acd1(4)*acd1(1)
      acd1(29)=acd1(6)*acd1(5)
      acd1(30)=-acd1(21)*acd1(20)
      acd1(31)=-acd1(23)*acd1(22)
      brack=acd1(24)+acd1(25)+acd1(26)+acd1(27)+acd1(28)+2.0_ki*acd1(29)+acd1(3&
      &0)+acd1(31)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p0_gg_gh_model
      use p0_gg_gh_kinematics
      use p0_gg_gh_color
      use p0_gg_gh_abbrevd1h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(19) :: acd1
      complex(ki) :: brack
      acd1(1)=d(iv1,iv2)
      acd1(2)=abb1(15)
      acd1(3)=k2(iv1)
      acd1(4)=spvak2k3(iv2)
      acd1(5)=abb1(23)
      acd1(6)=k2(iv2)
      acd1(7)=spvak2k3(iv1)
      acd1(8)=spvak1k2(iv2)
      acd1(9)=abb1(19)
      acd1(10)=spvak1k2(iv1)
      acd1(11)=spvak1k3(iv1)
      acd1(12)=spvak2k1(iv2)
      acd1(13)=abb1(11)
      acd1(14)=spvak1k3(iv2)
      acd1(15)=spvak2k1(iv1)
      acd1(16)=acd1(3)*acd1(4)
      acd1(17)=acd1(6)*acd1(7)
      acd1(16)=acd1(17)+acd1(16)
      acd1(16)=acd1(5)*acd1(16)
      acd1(17)=acd1(8)*acd1(7)
      acd1(18)=acd1(10)*acd1(4)
      acd1(17)=acd1(18)+acd1(17)
      acd1(17)=acd1(9)*acd1(17)
      acd1(18)=acd1(12)*acd1(11)
      acd1(19)=acd1(15)*acd1(14)
      acd1(18)=acd1(19)+acd1(18)
      acd1(18)=acd1(13)*acd1(18)
      acd1(19)=acd1(2)*acd1(1)
      brack=acd1(16)+acd1(17)+acd1(18)+2.0_ki*acd1(19)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p0_gg_gh_model
      use p0_gg_gh_kinematics
      use p0_gg_gh_color
      use p0_gg_gh_abbrevd1h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd1
      complex(ki) :: brack
      brack=0.0_ki
   end function brack_4
!---#] function brack_4:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3) result(numerator)
      use p0_gg_gh_globalsl1, only: epspow
      use p0_gg_gh_kinematics
      use p0_gg_gh_abbrevd1h0
      implicit none
      complex(ki), intent(in) :: mu2
      integer, intent(in), optional :: i1
      integer, intent(in), optional :: i2
      integer, intent(in), optional :: i3
      complex(ki) :: numerator
      complex(ki) :: loc
      integer :: t1
      integer :: deg
      complex(ki), dimension(4), parameter :: Q = (/ (0.0_ki,0.0_ki),(0.0_ki,0.&
      &0_ki),(0.0_ki,0.0_ki),(0.0_ki,0.0_ki)/)
      qshift = -k4
      numerator = 0.0_ki
      deg = 0
      if(present(i1)) then
          iv1=i1
          deg=1
      else
          iv1=1
      end if
      if(present(i2)) then
          iv2=i2
          deg=2
      else
          iv2=1
      end if
      if(present(i3)) then
          iv3=i3
          deg=3
      else
          iv3=1
      end if
      t1 = 0
      if(deg.eq.0) then
         numerator = cond(epspow.eq.t1,brack_1,Q,mu2)
         return
      end if
      if(deg.eq.1) then
         numerator = cond(epspow.eq.t1,brack_2,Q,mu2)
         return
      end if
      if(deg.eq.2) then
         numerator = cond(epspow.eq.t1,brack_3,Q,mu2)
         return
      end if
      if(deg.eq.3) then
         numerator = cond(epspow.eq.t1,brack_4,Q,mu2)
         return
      end if
   end function derivative
!---#] function derivative:
end module     p0_gg_gh_d1h0l1d
