module     p0_gg_gh_d1h1l1d
   ! file: /mt/home/sjones/repos/POWHEG-BOX/ggh-gosam/GoSam_POWHEG/Virtual/p0_g &
   ! &g_gh/helicity1d1h1l1d.f90
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
      use p0_gg_gh_abbrevd1h1
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(11) :: acd1
      complex(ki) :: brack
      acd1(1)=dotproduct(qshift,spvak2k1)
      acd1(2)=dotproduct(qshift,spvak2k3)
      acd1(3)=abb1(6)
      acd1(4)=abb1(7)
      acd1(5)=abb1(8)
      acd1(6)=dotproduct(qshift,spvak2l4)
      acd1(7)=abb1(9)
      acd1(8)=abb1(12)
      acd1(9)=acd1(3)*acd1(1)
      acd1(9)=-acd1(5)+acd1(9)
      acd1(9)=acd1(2)*acd1(9)
      acd1(10)=-acd1(4)*acd1(1)
      acd1(11)=-acd1(7)*acd1(6)
      brack=acd1(8)+acd1(9)+acd1(10)+acd1(11)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p0_gg_gh_model
      use p0_gg_gh_kinematics
      use p0_gg_gh_color
      use p0_gg_gh_abbrevd1h1
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(12) :: acd1
      complex(ki) :: brack
      acd1(1)=spvak2k1(iv1)
      acd1(2)=dotproduct(qshift,spvak2k3)
      acd1(3)=abb1(6)
      acd1(4)=abb1(7)
      acd1(5)=spvak2k3(iv1)
      acd1(6)=dotproduct(qshift,spvak2k1)
      acd1(7)=abb1(8)
      acd1(8)=spvak2l4(iv1)
      acd1(9)=abb1(9)
      acd1(10)=acd1(2)*acd1(3)
      acd1(10)=acd1(10)-acd1(4)
      acd1(10)=acd1(1)*acd1(10)
      acd1(11)=acd1(6)*acd1(3)
      acd1(11)=-acd1(7)+acd1(11)
      acd1(11)=acd1(5)*acd1(11)
      acd1(12)=-acd1(9)*acd1(8)
      brack=acd1(10)+acd1(11)+acd1(12)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p0_gg_gh_model
      use p0_gg_gh_kinematics
      use p0_gg_gh_color
      use p0_gg_gh_abbrevd1h1
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(7) :: acd1
      complex(ki) :: brack
      acd1(1)=spvak2k1(iv1)
      acd1(2)=spvak2k3(iv2)
      acd1(3)=abb1(6)
      acd1(4)=spvak2k1(iv2)
      acd1(5)=spvak2k3(iv1)
      acd1(6)=acd1(2)*acd1(1)
      acd1(7)=acd1(5)*acd1(4)
      acd1(6)=acd1(6)+acd1(7)
      brack=acd1(6)*acd1(3)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p0_gg_gh_model
      use p0_gg_gh_kinematics
      use p0_gg_gh_color
      use p0_gg_gh_abbrevd1h1
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
      use p0_gg_gh_abbrevd1h1
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
end module     p0_gg_gh_d1h1l1d
